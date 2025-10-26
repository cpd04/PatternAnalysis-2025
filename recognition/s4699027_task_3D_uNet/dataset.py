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

def load_prostate_data(data_file_path, train_data=1, test_data=1, train_split=0.7,
                       validate_split=0.15, batch_size=32):
    '''
    Load the prostate MRI data in the NIFTI format for training and testing. Data
    is augmented appropriately for better generalisation performance. 

    data_file_path : Path to the NIFTI data file folder
    train_data : Flag to load and export the training data set
    test_data : Flag to load and export the testing data set
    '''

    # Initialise train, validation, and test sets
    x_train = []
    x_validate = []
    x_test = []
    
    # Data augmentation (might also want to shuffle):
    # Flip data
    # Rotate data
    # Scale data
    # Normalise data (done in the import function)

    # Directory path information
    image_file_path = "/semantic_MRs_anon/"
    label_file_path = "/semantic_labels_anon/"

    data_image_directory = os.fsencode(data_file_path + image_file_path)
    data_label_directory = os.fsencode(data_file_path + label_file_path)

    # Retrieve all file names from directory
    image_list = []
    label_list = []

    for file in sorted(os.listdir(data_image_directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii.gz"):
            image_list.append(os.path.join(data_file_path + image_file_path, filename))

    for file in sorted(os.listdir(data_label_directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii.gz"):
            label_list.append(os.path.join(data_file_path + label_file_path, filename))

    # Load both data and labels
    print("Loading prostate MRI data...")
    images = torch.from_numpy(load_data_3D(image_list, normImage=True, 
                                           early_stop=True)).to(torch.float32)
    labels = torch.from_numpy(load_data_3D(label_list, categorical=True,
                                           dtype=np.uint8, early_stop=True)
                                           ).to(torch.uint8)

    # Add an additional channel dimension for labels for the format (N, C, D, H, W )
    images = images[:, torch.newaxis, :, :, :]
    labels = labels.permute(0, 4, 1, 2, 3)

    print(f"Loaded {len(images)} images and {len(labels)} labels successfully")

    # Split into training, validation, and testing sets
    num_total = len(images)
    num_train = int(0.7 * num_total)
    num_validate = int(0.15 * num_total)
    num_test = num_total - num_train - num_validate

    X_train_dataset = images[:num_train]
    Y_train_dataset = labels[:num_train]

    X_validate_dataset = images[num_train:num_train + num_validate]
    Y_validate_dataset = labels[num_train:num_train + num_validate]

    X_test_dataset = images[num_train + num_validate:]
    Y_test_dataset = labels[num_train + num_validate:]

    # Now make into datasets
    train_dataset = TensorDataset(X_train_dataset, Y_train_dataset)
    validation_dataset = TensorDataset(X_validate_dataset, Y_validate_dataset)
    test_dataset = TensorDataset(X_test_dataset, Y_test_dataset)

    # Loaders for training, validation, and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return [train_loader, validation_loader, test_loader]

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "data", "HipMRI_study_complete_release_v1")
    load_prostate_data(data_path, train_data=1, test_data=1)