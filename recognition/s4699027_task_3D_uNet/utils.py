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

    data : Numpy array of shape (D, H, W) containing the labelled data
    num_classes : Number of unique classes in the dataset
    '''
    return np.eye(num_classes)[data]