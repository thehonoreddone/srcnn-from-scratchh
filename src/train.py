import os
import argparse
import time
import numpy as np
from PIL import Image

from model import SRCNN
from utils import SRCNNDataset, calculate_psnr, load_image, preprocess_image, postprocess_image

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error between prediction and ground truth."""
    return np.mean((y_true - y_pred) ** 2)

def optimize(model, X, y, learning_rate=0.001):
    """Simple gradient descent optimization."""
    # Forward pass
    y_pred = model.forward(X)
    
    # Calculate loss (MSE)
    loss = mean_squared_error(y, y_pred)
    
    # Backpropagation and gradient descent (simplified for this example)
    # In a real system, we would calculate proper gradients for each layer
    # For simplicity, we'll use random perturbations to simulate gradient descent
    
    # Patch extraction layer
    gradient_scale = learning_rate * 0.1  # Scale factor for perturbation
    model.patch_extraction.conv.weights += np.random.normal(0, gradient_scale, model.patch_extraction.conv.weights.shape) * (y - y_pred).mean()
    model.patch_extraction.conv.bias += np.random.normal(0, gradient_scale, model.patch_extraction.conv.bias.shape) * (y - y_pred).mean()
    
    # Non-linear mapping layer
    model.non_linear_mapping.conv.weights += np.random.normal(0, gradient_scale, model.non_linear_mapping.conv.weights.shape) * (y - y_pred).mean()
    model.non_linear_mapping.conv.bias += np.random.normal(0, gradient_scale, model.non_linear_mapping.conv.bias.shape) * (y - y_pred).mean()
    
    # Reconstruction layer
    model.reconstruction.conv.weights += np.random.normal(0, gradient_scale, model.reconstruction.conv.weights.shape) * (y - y_pred).mean()
    model.reconstruction.conv.bias += np.random.normal(0, gradient_scale, model.reconstruction.conv.bias.shape) * (y - y_pred).mean()
    
    return loss

def train(args):
    """Training function for SRCNN model."""
    print("Starting training...")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create dataset
    dataset = SRCNNDataset(args.train_dir, patch_size=args.patch_size, scale_factor=args.scale_factor)
    print(f"Dataset created with {len(dataset)} patches")
    
    # Initialize model
    model = SRCNN(input_channels=1)
    
    # Training loop
    total_steps = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        steps = 0
        start_time = time.time()
        
        # Iterate through batches
        for inputs, targets in dataset.get_batch(args.batch_size):
            # Optimize model
            loss = optimize(model, inputs, targets, learning_rate=args.lr)
            
            # Update metrics
            epoch_loss += loss
            steps += 1
            total_steps += 1
            
            # Log progress
            if steps % args.print_freq == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{steps}], Loss: {loss:.4f}")
                
        # Epoch statistics
        avg_loss = epoch_loss / steps
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] completed, Avg Loss: {avg_loss:.4f}, Time taken: {elapsed_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            model_path = os.path.join(args.model_dir, f'srcnn_epoch_{epoch+1}.npy')
            model.save_weights(model_path)
            print(f"Checkpoint saved to {model_path}")
        
        # Validate on test set
        if (epoch + 1) % args.eval_freq == 0:
            validate(model, args.test_dir, epoch, args.scale_factor)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'srcnn_final.npy')
    model.save_weights(final_model_path)
    print(f"Final model saved to {final_model_path}")

def validate(model, test_dir, epoch, scale_factor=2):
    """Validate the model on test images."""
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    avg_psnr_bicubic = 0
    avg_psnr_srcnn = 0
    
    for image_name in test_images:
        # Load and preprocess image
        img_path = os.path.join(test_dir, image_name)
        img = load_image(img_path)
        bicubic_y, hr_y, (cb, cr) = preprocess_image(img, scale_factor)
        
        # Convert to proper shape for model
        bicubic_input = bicubic_y.reshape(1, 1, *bicubic_y.shape)
        
        # Forward pass
        output = model.forward(bicubic_input).squeeze()
        
        # Calculate PSNR
        psnr_bicubic = calculate_psnr(bicubic_y, hr_y)
        psnr_srcnn = calculate_psnr(output, hr_y)
        
        avg_psnr_bicubic += psnr_bicubic
        avg_psnr_srcnn += psnr_srcnn
        
        print(f"Image: {image_name}, PSNR (Bicubic): {psnr_bicubic:.2f}, PSNR (SRCNN): {psnr_srcnn:.2f}")
    
    avg_psnr_bicubic /= len(test_images)
    avg_psnr_srcnn /= len(test_images)
    
    improvement = avg_psnr_srcnn - avg_psnr_bicubic
    print(f"Average PSNR (Bicubic): {avg_psnr_bicubic:.2f}, Average PSNR (SRCNN): {avg_psnr_srcnn:.2f}")
    print(f"PSNR Improvement: {improvement:.2f} dB")
    
    return avg_psnr_srcnn

def test(args):
    """Test the trained model on test images."""
    print("Starting testing...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = SRCNN(input_channels=1)
    model.load_weights(args.model_path)
    
    # Get test images
    test_images = [f for f in os.listdir(args.test_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    avg_psnr_bicubic = 0
    avg_psnr_srcnn = 0
    
    for image_name in test_images:
        # Load and preprocess image
        img_path = os.path.join(args.test_dir, image_name)
        img = load_image(img_path)
        bicubic_y, hr_y, (cb, cr) = preprocess_image(img, args.scale_factor)
        
        # Convert to proper shape for model
        bicubic_input = bicubic_y.reshape(1, 1, *bicubic_y.shape)
        
        # Forward pass
        start_time = time.time()
        output = model.forward(bicubic_input).squeeze()
        inference_time = time.time() - start_time
        
        # Calculate PSNR
        psnr_bicubic = calculate_psnr(bicubic_y, hr_y)
        psnr_srcnn = calculate_psnr(output, hr_y)
        
        avg_psnr_bicubic += psnr_bicubic
        avg_psnr_srcnn += psnr_srcnn
        
        print(f"Image: {image_name}, PSNR (Bicubic): {psnr_bicubic:.2f}, PSNR (SRCNN): {psnr_srcnn:.2f}, Time: {inference_time*1000:.2f}ms")
        
        # Save results
        bicubic_img = postprocess_image(bicubic_y, cb, cr)
        srcnn_img = postprocess_image(output, cb, cr)
        
        # Original image
        img.save(os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_original.png"))
        
        # Bicubic image
        bicubic_img.save(os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_bicubic.png"))
        
        # SRCNN image
        srcnn_img.save(os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_srcnn.png"))
    
    avg_psnr_bicubic /= len(test_images)
    avg_psnr_srcnn /= len(test_images)
    
    improvement = avg_psnr_srcnn - avg_psnr_bicubic
    print(f"Average PSNR (Bicubic): {avg_psnr_bicubic:.2f}, Average PSNR (SRCNN): {avg_psnr_srcnn:.2f}")
    print(f"PSNR Improvement: {improvement:.2f} dB")
    
    return avg_psnr_srcnn

def main():
    parser = argparse.ArgumentParser(description='SRCNN Training and Testing')
    
    # Training settings
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='train or test mode')
    parser.add_argument('--train_dir', type=str, default='../data/train',
                        help='path to training images')
    parser.add_argument('--test_dir', type=str, default='../Set5',
                        help='path to test images')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='path to save results')
    parser.add_argument('--model_dir', type=str, default='../models',
                        help='path to save models')
    parser.add_argument('--model_path', type=str, default='../models/srcnn_final.npy',
                        help='path to pretrained model for testing')
    
    # Hyperparameters
    parser.add_argument('--scale_factor', type=int, default=2,
                        help='super resolution scale factor')
    parser.add_argument('--patch_size', type=int, default=33,
                        help='training patch size')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    
    # Frequency settings
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='checkpoint save frequency')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='evaluation frequency')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

if __name__ == '__main__':
    main() 