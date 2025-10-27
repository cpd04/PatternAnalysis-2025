#!/usr/bin/env python
"""
Script containing the modules used in the construction of a 3D Improved UNet 
Model. Each module is individually constructed and later combined to form the 
full architecture.

@author Connor Davis
"""

# Import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate the U Net model - Done with ChatGPT
import torch
import torch.nn as nn
import torch.nn.functional as F

NEGATIVE_SLOPE = 0.01

class ContextModule(nn.Module):
    """(IN => LeakyReLU => 3D Conv) * 2 + skip"""
    """Dropout layer in between conv layers"""
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        
        self.context = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),

            nn.Dropout3d(p=0.3), # Unsure on the effectiveness
            
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.context(x)

class UpsamplingModule(nn.Module):
    """(2x Upscale => ReLU => BN) * 2"""
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        
        self.action = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            # Concatenations
        )

    def forward(self, x):
        return self.action(x)

class LocalisationModule(nn.Module):
    """3D Conv (k=3) => LeakyReLU => 3d Conv (k=1) => LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()

        self.localisation = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(NEGATIVE_SLOPE),

            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

    def forward(self, x):
        return self.localisation(x)
    
class SegmentationLayer(nn.Module):
    """(Conv => ReLU => BN) * 2"""
    def __init__(self, in_channels, out_channels):
        super(SegmentationLayer, self).__init__()
        
        self.action = nn.Sequential(
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.action(x)

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_activation = nn.Softmax(dim=1)

        # Down path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                DoubleConv(feature*2, feature)
            )

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder (down)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for up path

        # Decoder (up)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # upsample
            skip = skip_connections[idx // 2]

            # In case input size isn't perfectly divisible
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)  # concatenate
            x = self.ups[idx+1](x)

        x = self.final_conv(x)
        return self.final_activation(x)  # Apply softmax activation for 4 categories
