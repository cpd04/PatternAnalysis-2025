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
DROP_OUT_PROB = 0.3

class ContextModule(nn.Module):
    """
    Context module consisting of two instance normalisation layers, two 
    LeakyReLU activations, and two 3D convolutional layers with a dropout layer
    in between the convolutional layers.
    """
    def __init__(self, in_channels, out_channels):
        """
       IN => LeakyReLU => 3D Conv (k=3) => Dropout => IN => LeakyReLU => 3D Conv (k=3)

        Arguments:
            in_channels : Number of input channels to the context module
            out_channels : Number of output channels from the context module
        """
            
        super(ContextModule, self).__init__()
        
        # Context layer step
        self.context = nn.Sequential(
            # First convolution block
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),

            # Improved with Dropout
            nn.Dropout3d(p=DROP_OUT_PROB), 
            
            # Second convolution block
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Forward pass of the context module.
        
        Arguments:
            x : Input tensor to the context module
        
        Returns:
            Output tensor from the context module
        """
        return self.context(x)

class UpsamplingModule(nn.Module):
    """
    Upsampling module consisting of an upsampling layer followed by a 3D 
    convolutional layer and a LeakyReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        """
        Upsampling => 3D Conv (k=3) => LeakyReLU

        Arguments:
            in_channels : Number of input channels to the upsampling module
            out_channels : Number of output channels from the upsampling module
        """
        super(UpsamplingModule, self).__init__()
        
        # Upsampling layer step
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

    def forward(self, x):
        """
        Forward pass of the upsampling module.
        
        Arguments:
            x : Input tensor to the upsampling module
        
        Returns:
            Output tensor from the upsampling module
        """
        return self.upsample(x)

class LocalisationModule(nn.Module):
    """
    Localisation module consisting of two 3D convolutional layers with a 
    LeakyReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        """
        3D Conv (k=3) => LeakyReLU => 3d Conv (k=1) => LeakyReLU
        
        Arguments:
            in_channels : Number of input channels to the localisation module
            out_channels : Number of output channels from the localisation module
        """
        super(LocalisationModule, self).__init__()

        # Localisation layer step
        self.localisation = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(NEGATIVE_SLOPE),

            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

    def forward(self, x):
        """
        Forward pass of the localisation module.
        
        Arguments:
            x : Input tensor to the localisation module
        
        Returns:
            Output tensor from the localisation module
        """
        return self.localisation(x)

class DecreaseLayer(nn.Module):
    """
    Decrease layer consisting of a 3D convolutional layer followed by a 
    LeakyReLU activation and a context module. Represents one step in the down
    sampling path of the 3D Improved UNet architecture.
    """
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        """
        3D Conv (k=3) => LeakyReLU => context module

        Arguments:
            in_channels : Number of input channels to the decrease layer
            out_channels : Number of output channels from the decrease layer
            stride : Stride for the convolutional layer
            padding : Padding for the convolutional layer
        """
        
        super(DecreaseLayer, self).__init__()
        
        # Decrease layer step
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.LeakyReLU(NEGATIVE_SLOPE),
        )

        self.context = ContextModule(out_channels, out_channels)

    def forward(self, x):
        """
        Forward pass of the decrease layer.
        
        Arguments:
            x : Input tensor to the decrease layer
        
        Returns:
            z : Output tensor from the decrease layer
        """
        
        intermediate = self.conv(x)
        z = self.context(intermediate) + intermediate
        return z

class SegmentationLayer(nn.Module):
    """
    Segmentation layer consisting of a 1x1x1 3D convolutional layer to reduce
    the number of channels to the desired output classes.
    """
    def __init__(self, in_channels, out_channels):
        """
        1x1x1 3D Conv
        
        Arguments:
            in_channels : Number of input channels to the segmentation layer
            out_channels : Number of output channels from the segmentation layer
        """
        super(SegmentationLayer, self).__init__()

        # 1 x 1 x 1 Stride-1 Convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the segmentation layer.
        
        Arguments:
            x : Input tensor to the segmentation layer
        
        Returns:
            Output tensor from the segmentation layer
        """
        
        return self.conv(x)

class ImprovedUNet(nn.Module):
    """
    Improved 3D UNet architecture for volumetric segmentation. Combines the
    previously defined modules to create the full network. Architecture is 
    based on "Brain Tumor Segmentation and Radiomics Survival Prediction: 
    Contribution to the BRATS 2017 Challenge" by Isensee et al. (2018).
    """
    def __init__(self, in_channels=1, out_channels=6, features=[32, 64, 128, 256]):
        """
        Constructs the Improved 3D UNet architecture.
        
        Arguments:
            in_channels : Number of input channels to the network
            out_channels : Number of output channels from the network
            features : List of feature sizes for each layer in the down path
        """
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
            in_channels = feature * 2
            self.localisation_layers.append(LocalisationModule(in_channels, feature))

        # Segmentation layers
        self.segmentation_layer_3 = SegmentationLayer(64, out_channels)
        self.segmentation_layer_2 = SegmentationLayer(32, out_channels)
        self.segmentation_layer_1 = SegmentationLayer(32, out_channels)
        
        self.segmentation_layer_3_upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.segmentation_layer_2_3_upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Final layers
        self.final_conv = nn.Conv3d(features[0], features[0], kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the Improved 3D UNet.

        Arguments:
            x : Input tensor to the network
        """
        
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
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.localisation_layers[1](x)
        x = self.upsampling_layers[1](x)

        # Layer 3 - Up Path
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.localisation_layers[2](x)
        seg_3 = self.segmentation_layer_3(x)
        x = self.upsampling_layers[2](x)

        # Layer 2 - Up Path
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.localisation_layers[3](x)
        seg_2 = self.segmentation_layer_2(x)
        x = self.upsampling_layers[3](x)

        # Layer 1 - Up Path
        x = torch.cat((skip_connections[4], x), dim=1)
        x = self.final_conv(x)
        seg_1 = self.segmentation_layer_1(x)

        # Upsample Segmentation Maps
        seg_3_upsampled = self.segmentation_layer_3_upsample(seg_3)
        seg_2_3 = seg_2 + seg_3_upsampled 
        seg_2_3_upsampled = self.segmentation_layer_2_3_upsample(seg_2_3)
        final_seg = seg_1 + seg_2_3_upsampled

        # Final Convolution and Softmax
        return self.final_activation(final_seg)