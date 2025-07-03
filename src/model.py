import numpy as np
from .layers import PatchExtraction, NonLinearMapping, Reconstruction

class SRCNN:
    """
    Super Resolution CNN (SRCNN) implementation based on the paper:
    "Image Super-Resolution Using Deep Convolutional Networks" by Dong et al.
    
    The network consists of three convolutional layers:
    1. Patch extraction and representation
    2. Non-linear mapping
    3. Reconstruction
    """
    
    def __init__(self, input_channels=1):
        self.patch_extraction = PatchExtraction(in_channels=input_channels, out_channels=64, kernel_size=9, padding=4)
        self.non_linear_mapping = NonLinearMapping(in_channels=64, out_channels=32, kernel_size=1)
        self.reconstruction = Reconstruction(in_channels=32, out_channels=input_channels, kernel_size=5, padding=2)
        
    def forward(self, x):
        """
        Forward pass of the SRCNN model.
        
        Args:
            x: Input array of shape (batch_size, channels, height, width)
               representing low-resolution images upscaled with bicubic interpolation
               
        Returns:
            Super-resolution output of the same size as input
        """
        x = self.patch_extraction.forward(x)
        x = self.non_linear_mapping.forward(x)
        x = self.reconstruction.forward(x)
        return x
    
    def save_weights(self, path):
        """Save model weights to a file."""
        weights = {
            'patch_extraction': {
                'weights': self.patch_extraction.conv.weights,
                'bias': self.patch_extraction.conv.bias
            },
            'non_linear_mapping': {
                'weights': self.non_linear_mapping.conv.weights,
                'bias': self.non_linear_mapping.conv.bias
            },
            'reconstruction': {
                'weights': self.reconstruction.conv.weights,
                'bias': self.reconstruction.conv.bias
            }
        }
        np.save(path, weights)
        
    def load_weights(self, path):
        """Load model weights from a file."""
        weights = np.load(path, allow_pickle=True).item()
        
        self.patch_extraction.conv.weights = weights['patch_extraction']['weights']
        self.patch_extraction.conv.bias = weights['patch_extraction']['bias']
        
        self.non_linear_mapping.conv.weights = weights['non_linear_mapping']['weights']
        self.non_linear_mapping.conv.bias = weights['non_linear_mapping']['bias']
        
        self.reconstruction.conv.weights = weights['reconstruction']['weights']
        self.reconstruction.conv.bias = weights['reconstruction']['bias'] 