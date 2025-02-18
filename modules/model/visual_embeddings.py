import torch
import torch.nn as nn

class VisualEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim=256):
        super(VisualEmbedding, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.projection = nn.Linear(in_channels, embedding_dim) # Linear Projection
        self.activation = nn.ReLU() # ReLU Activation
    
    def forward(self, x):
        x = self.global_avg_pool(x) # reduce the spatial dimensions
        x = torch.flatten(x, 1) # flatten the tensor
        x = self.projection(x) # Linear transformation
        x = self.activation(x) # ReLU activation
        return x
    
    