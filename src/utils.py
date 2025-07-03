import os
import numpy as np
from PIL import Image

def load_image(image_path):
    """Load an image and convert it to RGB format."""
    img = Image.open(image_path).convert('RGB')
    return img

def preprocess_image(img, scale_factor=2):
    """Prepare image for SRCNN processing:
    1. Convert to YCbCr color space
    2. Extract Y channel (luminance)
    3. Downscale and upscale to create low-resolution input
    """
    # Convert to YCbCr and extract Y channel
    img_ycbcr = img.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()
    
    # Convert Y to numpy array
    y_np = np.array(y).astype(np.float32) / 255.0
    
    # Create low-res version
    h, w = y_np.shape
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # Saf Python ile bicubic downscale
    lr_y = bicubic_resize(y_np, (lr_h, lr_w))
    
    # Saf Python ile bicubic upscale
    bicubic_y = bicubic_resize(lr_y, (h, w))
    
    # Original Y will be the ground truth (target)
    return bicubic_y, y_np, (cb, cr)

def postprocess_image(y_pred, cb, cr):
    """Convert Y channel back to RGB image."""
    # Clip values to [0, 1] and convert to uint8
    y_pred = np.clip(y_pred, 0, 1) * 255
    y_pred = y_pred.astype(np.uint8)
    
    # Merge with original Cb and Cr channels
    y_pred = Image.fromarray(y_pred, mode='L')
    result = Image.merge('YCbCr', [y_pred, cb, cr]).convert('RGB')
    
    return result

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def bicubic_kernel(x):
    """Bicubic interpolation kernel."""
    x = abs(x)
    if x <= 1:
        return 1.5 * x**3 - 2.5 * x**2 + 1
    elif x < 2:
        return -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
    else:
        return 0

def bicubic_resize(img, size):
    """Simple bicubic interpolation implementation."""
    src_h, src_w = img.shape
    dst_h, dst_w = size
    
    # Scale factors
    scale_h = src_h / dst_h
    scale_w = src_w / dst_w
    
    # Create output image
    dst = np.zeros((dst_h, dst_w), dtype=np.float32)
    
    # Bicubic interpolation
    for y in range(dst_h):
        for x in range(dst_w):
            # Source position
            src_x = (x + 0.5) * scale_w - 0.5
            src_y = (y + 0.5) * scale_h - 0.5
            
            # Get the surrounding 4x4 pixels
            x_start = int(src_x) - 1
            y_start = int(src_y) - 1
            
            # Calculate weights
            total_weight = 0
            value = 0
            
            for i in range(4):
                y_pos = y_start + i
                # Clamp y position
                y_pos = max(0, min(src_h - 1, y_pos))
                
                for j in range(4):
                    x_pos = x_start + j
                    # Clamp x position
                    x_pos = max(0, min(src_w - 1, x_pos))
                    
                    # Calculate weights using bicubic kernel
                    dx = src_x - x_pos
                    dy = src_y - y_pos
                    weight = bicubic_kernel(dx) * bicubic_kernel(dy)
                    
                    # Accumulate weighted values
                    value += img[y_pos, x_pos] * weight
                    total_weight += weight
            
            # Normalize and set output pixel
            if total_weight > 0:
                dst[y, x] = value / total_weight
            else:
                dst[y, x] = 0
                
    return dst

class SRCNNDataset:
    def __init__(self, image_dir, patch_size=33, scale_factor=2):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.patches = []
        self._create_patches()
        
    def _create_patches(self):
        for file in self.image_files:
            img_path = os.path.join(self.image_dir, file)
            img = load_image(img_path)
            bicubic_y, hr_y, _ = preprocess_image(img, self.scale_factor)
            
            h, w = bicubic_y.shape
            
            # Create patches by sliding window
            for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
                for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                    bicubic_patch = bicubic_y[i:i+self.patch_size, j:j+self.patch_size]
                    hr_patch = hr_y[i:i+self.patch_size, j:j+self.patch_size]
                    
                    # Reshape for SRCNN input format (N, C, H, W)
                    bicubic_patch = bicubic_patch.reshape(1, 1, self.patch_size, self.patch_size)
                    hr_patch = hr_patch.reshape(1, 1, self.patch_size, self.patch_size)
                    
                    self.patches.append((bicubic_patch, hr_patch))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx]
        
    def get_batch(self, batch_size, shuffle=True):
        """Get a batch of patches"""
        n_samples = len(self.patches)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Initialize batch arrays
            batch_inputs = None
            batch_targets = None
            
            for i, idx in enumerate(batch_indices):
                input_patch, target_patch = self.patches[idx]
                
                # For the first item, create the batch arrays
                if i == 0:
                    batch_inputs = np.zeros((len(batch_indices), *input_patch.shape[1:]), dtype=input_patch.dtype)
                    batch_targets = np.zeros((len(batch_indices), *target_patch.shape[1:]), dtype=target_patch.dtype)
                
                # Add to batch
                batch_inputs[i] = input_patch
                batch_targets[i] = target_patch
                
            yield batch_inputs, batch_targets 