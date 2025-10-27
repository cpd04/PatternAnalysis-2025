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

NEGATIVE_SLOPE = 0.01

class ContextModule(nn.Module):
    """(IN => LeakyReLU => 3D Conv) * 2 + skip"""
    """Dropout layer in between conv layers"""
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        
        # Context layer step
        self.context = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),

            nn.Dropout3d(p=0.3), # Unsure on the effectiveness, revisit
            
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.context(x)

class UpsamplingModule(nn.Module):
    """(2x Upscale => 3D Conv (k=3) => LeakyReLU) """
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        
        # Upsampling layer step
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

    def forward(self, x):
        return self.upsample(x)

class LocalisationModule(nn.Module):
    """3D Conv (k=3) => LeakyReLU => 3d Conv (k=1) => LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()

        # Localisation layer step
        self.localisation = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(NEGATIVE_SLOPE),

            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

    def forward(self, x):
        return self.localisation(x)

class DecreaseLayer(nn.Module):
    """3D Conv (k=3) => LeakyReLU => context module => skip module """
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(DecreaseLayer, self).__init__()
        
        # Decrease layer step
        self.step1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

        self.context = ContextModule(out_channels, out_channels)

    def forward(self, x):
        intermediate = self.step1(x)
        z = self.context(intermediate) + intermediate
        return z

class SegmentationLayer(nn.Module):
    """(Conv => ReLU => BN) * 2"""
    def __init__(self, in_channels, out_channels):
        super(SegmentationLayer, self).__init__()

        # 1 x 1 x 1 Stride-1 Convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, features=[32, 64, 128, 256]):
        super(ImprovedUNet, self).__init__()

        # Set a list of down connection features
        self.down_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.localisation_layers = nn.ModuleList()

        # Set the initial input of the down connection
        self.down_layers.append(DecreaseLayer(in_channels, 16, stride=1, padding=1))
        in_channels = 16

        # Down path
        for feature in features:
            self.down_layers.append(DecreaseLayer(in_channels, feature, stride=2, padding=1))
            in_channels = feature

        # Upsampling layers
        for feature in reversed(features):
            self.upsampling_layers.append(UpsamplingModule(in_channels, int(feature/2)))
            in_channels = int(feature/2)

        # Localisation layers
        for feature in reversed(features):
            self.localisation_layers.append(LocalisationModule(feature, int(feature/2)))

        # Segmentation layers
        self.segmentation_layer_3 = SegmentationLayer(64, 6)
        self.segmentation_layer_2 = SegmentationLayer(32, 6)
        self.segmentation_layer_1 = SegmentationLayer(32, 6)
        
        self.segmentation_layer_3_upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Final layers
        self.final_conv = nn.Conv3d(features[0], features[0], kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        skip_connections = []

        # Encoder (down)
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)

        # The current x is now the bottle neck value
        skip_connections = skip_connections[::-1]  # reverse for up path

        # Layer 5 - Up Path
        # Upsampling once
        x = self.upsampling_layers[0](x)

        # Layer 4 - Up Path
        # Concatenate, localisation module, then upsampling
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.localisation_layers[1](x)
        x = self.upsampling_layers[1](x)

        # Layer 3 - Up Path
        # Concatenate, localisation module, then segmentation layer
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.localisation_layers[2](x)
        seg_3 = self.segmentation_layer_3(x)
        seg_3_upsampled = self.segmentation_layer_3_upsample(seg_3)
        x = self.upsampling_layers[2](x)

        # Layer 2 - Up Path
        # Concatenate, localisation module, then segmentation layer
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.localisation_layers[3](x)
        seg_2 = self.segmentation_layer_2(x)
        x = self.upsampling_layers[3](x)

        # Layer 1 - Up Path
        # Concatenate, convolution
        x = torch.cat((skip_connections[4], x), dim=1)
        x = self.final_conv(x)
        seg_1 = self.segmentation_layer_1(x)

        # Final Convolution and Softmax
        outputs = seg_3_upsampled + seg_2 + seg_1
        return self.final_activation(outputs)
