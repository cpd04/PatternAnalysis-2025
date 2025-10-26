#!/usr/bin/env python
"""
Script for key utility functions, primarily the conversion of label arrays to
one-hot encoding.

@author Connor Davis
"""

import torch
import torch.nn.functional as F
import numpy as np

def to_channels(data, num_classes=6):
    '''
    Convert the labelled dataset to one-hot encoding along a new channel axis.

    dataset : Numpy array of shape (N, D, H, W) containing the labelled data
    num_classes : Number of unique classes in the dataset
    '''

    if data.ndim != 3:
        return data  # add batch dimension

    # First convert to tensor 
    data_tensor = torch.from_numpy(data).to(torch.long) # Ensure integer type for indexing
    
    # Apply one-hot encoding and re-arrange dimensions to (N, C, D, H, W)
    one_hot_encoded = F.one_hot(data_tensor, num_classes=num_classes)

    return one_hot_encoded