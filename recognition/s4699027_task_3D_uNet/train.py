#!/usr/bin/env python
"""
Script to train, validate, and test on data obtained from dataset.py for the 
3D Improved UNet. Model based on architecture found under modules.py. A 
variety of accuracy metrics (loss, accuracy, and multiclass dice and dice 
similarity coefficients) are used to evaluate the performance of the 
segmentation model, with visualisations of accuracy produced for clarity. The 
model and it's results are saved and logged for comparison.

@author Connor Davis
"""

# TODO: Implement ability to save model checkpoints during training

# Incoporate WAND during training to get good visualisation of errors
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import matplotlib.pyplot as plt

from modules import ImprovedUNet
from dataset import load_prostate_data

# Parameters for Training
BATCH_SIZE=4
EPOCHS=50
LEARNING_RATE=5e-4
MODEL_PATH = os.path.join("./models/", f"model_checkpoint.pth")

class SoftDiceLoss(nn.Module):
    # Needs to be updated with the dice loss used in the paper
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        probs: [B, C, D, H, W] raw network outputs (softmax applied)
        targets: [B, C, D, H, W] one-hot ground truth
        """
        # UNTESTED WITH 3D DATA, VERIFY FUNCTIONALITY

        # Flatten batch and spatial dimensions
        probs = probs.contiguous().view(probs.shape[0], probs.shape[1], -1)
        targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)

        # Compute intersection and union
        intersection = (probs * targets).sum(dim=2)
        union = probs.sum(dim=2) + targets.sum(dim=2)

        # Compute dice score per class and batch
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = -dice.mean()  # mean over batch and classes

        return dice_loss


def train_model():
    """
    Load the trained model and testing data to perform predictions. Visualisations
    are created and saved for analysis.
    """
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    data_path = os.path.join(os.path.dirname(__file__), "data", "HipMRI_study_complete_release_v1")
    train_loader, validation_loader, _ = load_prostate_data(data_path, 
                                                            train_data=1, 
                                                            validation_data=1, 
                                                            test_data=0, 
                                                            train_split=0.01, 
                                                            validation_split=0.01,
                                                            batch_size=BATCH_SIZE,
                                                            debugging_mode=True)
    
    # Initialize the model, loss function, and optimizer
    model = ImprovedUNet(in_channels=1, out_channels=6).to(device, dtype=torch.float16)
    criterion = SoftDiceLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params} trainable parameters.")

    # Training loop
    print("Training uNet...")

    for epoch in range(EPOCHS):
        # Set the model to training mode
        model.train()
        total_loss = 0

        # Train over batches
        for raw, segmented in train_loader:
            raw = raw.to(device, dtype=torch.float16)
            segmented = segmented.to(device, dtype=torch.float16)

            # Forward pass
            outputs = model(raw)
            loss = criterion(outputs, segmented)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * raw.size(0)  # accumulate loss over the batch size

        avg_loss = total_loss / len(train_loader.dataset)

        # Find the validation loss
        val_loss = None
        if validation_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_raw, val_segmented in validation_loader:
                    val_raw = val_raw.to(device, dtype=torch.float16)
                    val_segmented = val_segmented.to(device, dtype=torch.float16)

                    val_outputs = model(val_raw)
                    v_loss = loss(val_outputs, val_segmented)
                    val_loss += v_loss.item() * val_raw.size(0)

            val_loss /= len(validation_loader.dataset)

        if val_loss is not None:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Global Epoch {global_epoch+1}")
            global_val_loss = val_loss
        else:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}")

        global_epoch += 1
    
    return model

if __name__ == "__main__":
    train_model()