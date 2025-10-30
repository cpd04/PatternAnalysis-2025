#!/usr/bin/env python
"""
Script to import and load the training, testing, and validation data. Labelled 
datasets are developed in the Nifti file format.

@author Connor Davis
"""

import torch
import numpy as np
import os
import nibabel as nib
import torchvision.transforms as transforms
import utils

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    '''
    Convert a label array to one-hot encoding along a new last axis.
    '''
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    
    return res

def load_data_3D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, orient=False,
                 early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage : bool (normalise the image 0.0–1.0)
    orient : Apply orientation and resample image? Good for images with large slice
             thickness or anisotropic resolution
    dtype : Type of the data. If dtype = np.uint8, it is assumed that the data is labels
    early_stop : Stop loading pre-maturely? Leaves arrays mostly empty, for quick
                 loading and testing scripts.
    '''
    affines = []
    
    # ~ interp = ' continuous '
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = utils.im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        # ~ testResultName = " oriented . nii . gz "
        # ~ niftiImage . to_filename ( testResultName )

    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = utils.im.applyOrientation(niftiImage, interpolation=interp, scale=1)

        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data

        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = utils.to_channels(inImage)
            # ~ images [i ,: ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage #with Pad
        else:
            # ~ images [i ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage #with Pad

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def create_loader(image_list, label_list, batch_size=4, reduced_shape=False, shuffle=True):
    '''
    Load specific data for testing purposes.
    '''
    
    # Begin loading of data    
    images = torch.from_numpy(load_data_3D(image_list, normImage=True,
                                           dtype=np.float16)).to(torch.float16)
    labels = torch.from_numpy(load_data_3D(label_list, categorical=True,
                                           dtype=np.uint8)).to(torch.uint8)

    # Add an additional channel dimension for labels for the format (N, C, D, H, W )
    images = images[:, torch.newaxis, :, :, :]
    labels = labels.permute(0, 4, 1, 2, 3)

    # Only take the middle 64, 64, 32 variables
    if reduced_shape:
        images = images[:, :, 128-16:128+16, 128-16:128+16, 64-8:64+8] 
        labels = labels[:, :, 128-16:128+16, 128-16:128+16, 64-8:64+8]

    print(f"Loaded {len(images)} images and {len(labels)} labels successfully")
    print(f"Image tensor shape: {images.shape}, Label tensor shape: {labels.shape}")

    # Develop the data loader
    num_total = len(images) 
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, 
                              shuffle=shuffle)
    
    return loader

def load_prostate_data(data_file_path, train_data=1, validation_data=1, 
                       test_data=1, train_split=0.7, validation_split=0.15,
                        batch_size=4, debugging_mode=False):
    '''
    Load the prostate MRI data in the NIFTI format for training and testing. Data
    is augmented appropriately for better generalisation performance. 

    data_file_path : Path to the NIFTI data file folder
    train_data : Flag to load and export the training data set
    test_data : Flag to load and export the testing data set
    '''

    # Initialise train, validation, and test sets
    x_train_names = []
    y_train_names = []
    x_validate_names = []
    y_validate_names = []
    x_test_names = []
    y_test_names = []
    
    # Data augmentation (might also want to shuffle):
    # Flip data
    # Rotate data

    # Directory path information
    image_file_path = "/semantic_MRs_anon/"
    label_file_path = "/semantic_labels_anon/"

    data_image_directory = os.fsencode(data_file_path + image_file_path)
    data_label_directory = os.fsencode(data_file_path + label_file_path)

    # Retrieve all file names from directory
    image_list = []
    label_list = []

    # Load all file names
    for file in sorted(os.listdir(data_image_directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii.gz"):
            image_list.append(os.path.join(data_file_path + image_file_path, filename))

    for file in sorted(os.listdir(data_label_directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii.gz"):
            label_list.append(os.path.join(data_file_path + label_file_path, filename))
    
    # Load the data into memory
    loaders = [None, None, None]

    # Generate each of the seperate data loaders
    if train_data:
        print("Loading training data...")
        x_train_names.extend(image_list[:int(train_split * len(image_list))])
        y_train_names.extend(label_list[:int(train_split * len(label_list))])
        train_loader = create_loader(x_train_names, y_train_names, 
                                     batch_size=batch_size, reduced_shape=debugging_mode,
                                     shuffle=True)
        loaders[0] = train_loader

    if validation_data:
        print("Loading validation data...")
        x_validate_names.extend(image_list[int(train_split * len(image_list)):int((train_split + validation_split) * len(image_list))])
        y_validate_names.extend(label_list[int(train_split * len(label_list)):int((train_split + validation_split) * len(label_list))])
        validation_data_loader = create_loader(x_validate_names, 
                                               y_validate_names, 
                                               batch_size=batch_size, 
                                               reduced_shape=debugging_mode, 
                                               shuffle=True)
        loaders[1] = validation_data_loader

    if test_data:
        print("Loading testing data...")
        x_test_names.extend(image_list[int((train_split + validation_split) * len(image_list)):])
        y_test_names.extend(label_list[int((train_split + validation_split) * len(label_list)):])
        test_data_loader = create_loader(x_test_names, y_test_names, 
                                         batch_size=batch_size, reduced_shape=debugging_mode,
                                         shuffle=False)
        loaders[2] = test_data_loader

    return loaders    