# A reimplementation of ResNet with HDC
# from Understanding Convolution for Semantic Segmentation
# https://doi.org/10.1109/WACV.2018.00163

import torch.nn as nn

"""
Basic Units
"""

# Convolution layer + batch norm layer
def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels)
    )

# Convolution layer + batch norm layer + activation (ReLU)
def conv_bn_ac(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

# Downsampling layer
# The code doesn't seem to downsample the intermediate features, but the paper says they downsampled
# up to 8x
def downsample(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels),
    )

"""
Residual Blocks
"""

class ResidualHDCBlock_O(nn.Module):
    def __init__(self, in_channels, num_1x1a, num_3x3b, num_1x1c, dilation, downsample_rate=1):
        super().__init__()
        
        if (downsample_rate != 1):
            self.downsample = downsample(in_channels, in_channels, downsample_rate)

        # Residual path
        self.res_1x1 = conv_bn(in_channels=in_channels, out_channels=num_1x1c, kernel_size=1)
        
        # Main path
        self.conv_1x1_reduce = conv_bn_ac(in_channels=in_channels, out_channels=num_1x1a, kernel_size=1)

        self.conv_3x3 = conv_bn_ac(in_channels=num_1x1a, out_channels=num_3x3b, kernel_size=3, 
                                   padding=dilation, dilation=dilation, stride=downsample_rate)
        
        self.conv_1x1_increase = conv_bn_ac(in_channels=num_3x3b, out_channels=num_1x1c, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(residual)
        residual = self.res_1x1(residual)

        conv = self.conv_1x1_reduce(x)
        conv = self.conv_3x3(conv)
        conv = self.conv_1x1_increase(conv)

        out = conv + residual
        return self.relu(out)


class ResidualHDCBlock_X(nn.Module):
    def __init__(self, in_channels, num_1x1a, num_3x3b, num_1x1c, dilation):
        super().__init__()
        
        # Main path
        self.conv_1x1_reduce = conv_bn_ac(in_channels=in_channels, out_channels=num_1x1a, kernel_size=1)

        self.conv_3x3 = conv_bn_ac(in_channels=num_1x1a, out_channels=num_3x3b, kernel_size=3, 
                                   padding=dilation, dilation=dilation)
        
        self.conv_1x1_increase = conv_bn_ac(in_channels=num_3x3b, out_channels=num_1x1c, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        conv = self.conv_1x1_reduce(x)
        conv = self.conv_3x3(conv)
        conv = self.conv_1x1_increase(conv)

        out = conv + residual

        return self.relu(out)


class ResidualHDCBlock_D(nn.Module):
    def __init__(self, in_channels, num_1x1a, num_3x3b, num_1x1c):
        super().__init__()
        
        # Residual path
        self.res_1x1 = conv_bn(in_channels=in_channels, out_channels=num_1x1c, kernel_size=1, stride=2)
        
        # Main path
        self.conv_1x1_reduce = conv_bn_ac(in_channels=in_channels, out_channels=num_1x1a, kernel_size=1, stride=2)

        self.conv_3x3 = conv_bn_ac(in_channels=num_1x1a, out_channels=num_3x3b, kernel_size=3, pad=1)
        
        self.conv_1x1_increase = conv_bn_ac(in_channels=num_3x3b, out_channels=num_1x1c, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        residual = self.res_1x1(residual)

        conv = self.conv_1x1_reduce(x)
        conv = self.conv_3x3(conv)
        conv = self.conv_1x1_increase(conv)

        out = conv + residual
        return self.relu(out)
    

"""
Resnet18-HDC
"""

class ResNet18_HDC(nn.Module):
    def __init__(self, num_classes=10, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.layers = \
        [
            conv_bn_ac(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            conv_bn_ac(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            conv_bn_ac(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),

            ResidualHDCBlock_O(64, 64, 64, 64, 1),
            ResidualHDCBlock_X(64, 64, 64, 64, 1),

            ResidualHDCBlock_O(64, 64, 64, 128, 1, downsample_rate=2),
            ResidualHDCBlock_X(128, 128, 128, 128, 1),

            ResidualHDCBlock_O(128, 128, 128, 256, 2, downsample_rate=2),
            ResidualHDCBlock_X(256, 256, 256, 256, 5),

            ResidualHDCBlock_O(256, 256, 256, 512, 5, downsample_rate=2),
            ResidualHDCBlock_X(512, 512, 512, 512, 9),

            nn.AvgPool2d(16, stride=1),   
        ]

        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i+1}", layer)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        for layer in self.layers:
            self._print(x.size())
            x = layer(x)

        self._print(x.size())
        x = x.view(x.size(0), -1)
        self._print(x.size())
        x = self.fc(x)

        return x
    
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)



"""
Resnet101-HDC
"""

class ResNet101_HDC(nn.Module):
    def __init__(self, num_classes=10, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.layers = \
        [
            conv_bn_ac(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            conv_bn_ac(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            conv_bn_ac(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),

            ResidualHDCBlock_O(64, 64, 64, 256, 1),
            ResidualHDCBlock_X(256, 64, 64, 256, 1),
            ResidualHDCBlock_X(256, 64, 64, 256, 1),

            ResidualHDCBlock_O(256, 128, 128, 512, 1, downsample_rate=2),
            ResidualHDCBlock_X(512, 128, 128, 512, 1),
            ResidualHDCBlock_X(512, 128, 128, 512, 1),
            ResidualHDCBlock_X(512, 128, 128, 512, 1),

            ResidualHDCBlock_O(512, 256, 256, 1024, 2, downsample_rate=2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 5),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 9),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 1),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 5),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 9),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 1),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 5),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 9),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 1),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 5),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 9),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 1),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 5),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 9),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 1),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 2),
            ResidualHDCBlock_X(1024, 256, 256, 1024, 5),  # 23 blocks in total

            ResidualHDCBlock_O(1024, 512, 512, 2048, 5, downsample_rate=2),
            ResidualHDCBlock_X(2048, 512, 512, 2048, 9),
            ResidualHDCBlock_X(2048, 512, 512, 2048, 17),

            nn.AvgPool2d(16, stride=1),   
        ]

        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i+1}", layer)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        for layer in self.layers:
            self._print(x.size())
            x = layer(x)

        self._print(x.size())
        x = x.view(x.size(0), -1)
        self._print(x.size())
        x = self.fc(x)

        return x
    
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)