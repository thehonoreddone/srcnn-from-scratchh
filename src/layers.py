import numpy as np

class ConvLayer:
    """Custom convolutional layer implementation with NumPy"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias using normal distribution
        self.weights = np.random.normal(0, 0.001, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        
    def forward(self, x):
        """
        Forward pass of convolutional layer
        
        Args:
            x: Input of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, _, h, w = x.shape
        
        # Pad input if necessary
        if self.padding > 0:
            x_padded = np.zeros((batch_size, self.in_channels, h + 2 * self.padding, w + 2 * self.padding))
            x_padded[:, :, self.padding:h+self.padding, self.padding:w+self.padding] = x
        else:
            x_padded = x
        
        # Calculate output dimensions
        h_padded, w_padded = x_padded.shape[2], x_padded.shape[3]
        h_out = (h_padded - self.kernel_size) // self.stride + 1
        w_out = (w_padded - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, h_out, w_out))
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(0, h_out):
                    for j in range(0, w_out):
                        # Extract patch
                        i_pos = i * self.stride
                        j_pos = j * self.stride
                        patch = x_padded[b, :, i_pos:i_pos+self.kernel_size, j_pos:j_pos+self.kernel_size]
                        
                        # Apply filter
                        output[b, c_out, i, j] = np.sum(self.weights[c_out] * patch) + self.bias[c_out]
        
        return output

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

class PatchExtraction:
    """First layer of SRCNN - Patch extraction and representation."""
    def __init__(self, in_channels=1, out_channels=64, kernel_size=9, padding=4):
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, padding=padding)
        
    def forward(self, x):
        return relu(self.conv.forward(x))
    
class NonLinearMapping:
    """Second layer of SRCNN - Non-linear mapping."""
    def __init__(self, in_channels=64, out_channels=32, kernel_size=1, padding=0):
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, padding=padding)
        
    def forward(self, x):
        return relu(self.conv.forward(x))
    
class Reconstruction:
    """Third layer of SRCNN - Reconstruction."""
    def __init__(self, in_channels=32, out_channels=1, kernel_size=5, padding=2):
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, padding=padding)
        
    def forward(self, x):
        return self.conv.forward(x)
