#!/usr/bin/env python
"""
Script for key utility functions, primarily the conversion of label arrays to
one-hot encoding.

@author Connor Davis
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# GUI Libraries for visualisation
import matplotlib
import imageio.v2 as imageio

def to_channels(data, num_classes=6):
    '''
    Convert the labelled dataset to one-hot encoding along a new channel axis.

    data : Numpy array of shape (D, H, W) containing the labelled data
    num_classes : Number of unique classes in the dataset
    '''
    return np.eye(num_classes)[data]

def to_single_channel(data):
    '''
    Convert one-hot encoded data back to single channel format.

    data : Numpy array of shape (C, D, H, W) containing one-hot encoded data
    '''
    # Convert all channels to single channel by taking argmax, then normalise
    single_channel = torch.argmax(data, dim=0, keepdim=True)
    norm_image = (single_channel / single_channel.max()).to(torch.float16)

    return norm_image

def generate_gif(volume, out_path, fps=16, cmap_name="viridis"):
    """
    Generate an animated GIF from a 3D NumPy array (volume).
    Each frame corresponds to one slice along the first axis.

    volume : 3D numpy array (C, D, H, W) -> (1, 256, 256, 128)
    """
    # Remove channel dimension
    volume = volume[0]  
    # Normalise to 0-255
    volume_norm = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)

    # Select a colormap
    cmap = matplotlib.colormaps[cmap_name]

    # Generate frames
    frames = []
    for i in range(volume.shape[2]):
        # Convert grayscale slice to RGBA using the colormap
        colored = (cmap(volume_norm[:, :, i] / 255.0)[:, :, :3] * 255).astype(np.uint8)
        frames.append(colored)
    
    # Save as animated GIF
    imageio.mimsave(out_path, frames, fps=fps, loop=0)

def combine_gifs(raw_path, true_path, pred_path, out_path="combined.gif", fps=50):
    # Read both GIFs as lists of frames (NumPy arrays)
    raw_model = imageio.mimread(raw_path)
    true_seg = imageio.mimread(true_path)
    pred_seg = imageio.mimread(pred_path)

    # Make sure they have the same number of frames
    n_frames = min(len(raw_model), len(true_seg), len(pred_seg))

    raw_model = raw_model[:n_frames]
    true_seg = true_seg[:n_frames]
    pred_seg = pred_seg[:n_frames]

    frames = []
    for f1, f2, f3 in zip(raw_model, true_seg, pred_seg):
        # Combine side by side
        combined = np.concatenate((f1, f2, f3), axis=1)
        frames.append(combined)

    # Save as new animated GIF
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
