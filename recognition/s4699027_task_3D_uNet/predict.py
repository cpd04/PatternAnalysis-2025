#!/usr/bin/env python
"""
Script to load both the trained 3D Improved UNet Model and the testing data to 
perform predictions. Visualisations for the inputs to the model, the true 
segmented image, and the predicted segmented image are constructed and saved.

@author Connor Davis
"""

import torch
from torch.functional import F
from torch.utils.data import DataLoader
import utils
import os

def dice_score(pred, target, num_classes: int = 6, epsilon: float = 1e-6):
    """
    Compute Dice coefficient per class.

    Args:
        pred: predicted masks (N, C, H, W) with class indices
        target: ground truth masks (N, C, H, W) with class indices
        num_classes: number of segmentation classes
        epsilon: smoothing term to avoid division by zero

    Returns:
        list of dice scores per class
    """

    # Ensure target is in float form
    target_one_hot = target.float()

    # Convert from probabilities to selected values
    pred = torch.argmax(pred, dim=1)  # [N, H, W, D]
    pred = F.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3).float().contiguous()  # [N, C, H, W, D]

    # Compute per-class Dice
    dims = (0, 2, 3, 4)  # sum over batch + spatial dims
    intersection = torch.sum(pred * target_one_hot, dims)
    cardinality = torch.sum(pred + target_one_hot, dims)

    dice_per_class = (2. * intersection + epsilon) / (cardinality + epsilon)
    return dice_per_class  # shape [C]


def evaluate_unet(model, loader: DataLoader, device: str = "cuda"):
    """
    Evaluate a trained U-Net on a test set using DSC per class.

    Args:
        model: trained UNet model
        test_loader: DataLoader for test set
        device: device string ("cuda" or "cpu")
    """
    model.eval()
    model.to(device)

    num_classes = 6
    total_dsc = [0.0 for _ in range(num_classes)]
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            # forward pass
            outputs = model(images)              # (N, C, H, W)

            dsc_batch = dice_score(outputs, masks, num_classes=num_classes)
            total_dsc = [t + d for t, d in zip(total_dsc, dsc_batch)]
            count += 1

    avg_dsc = [t / count for t in total_dsc]

    print("Dice per class:")
    for cls, dsc in enumerate(avg_dsc):
        print(f"  Class {cls}: {dsc:.4f}")

    return avg_dsc

def predict_single_image(model, loader: DataLoader, data_path: str, device: str = "cuda"):
    """
    Predict segmentation mask for a single image using the trained model.

    Args:
        model: trained UNet model
        image: input image tensor (1, C, H, W)
        device: device string ("cuda" or "cpu")

    Returns:
        predicted segmentation mask (1, C, H, W)
    """
    model.eval()
    model.to(device)

    raw, segmented = next(iter(loader))
    with torch.no_grad():
        # Forward pass prediction
        outputs = model(raw)              # (N, C, H, W)

        print(raw.shape)
        print(segmented.shape)
        print(outputs.shape)

        # Convert to single channel for visualisation
        segmented_single = utils.to_single_channel(segmented[0]).cpu().numpy()
        pred_single = utils.to_single_channel(outputs[0]).cpu().numpy()
        raw_single = raw[0].cpu().numpy()

        # Generate GIFs for visualisation
        raw_gif_path = os.path.join(os.path.dirname(__file__), data_path, "raw.gif")
        segmented_gif_path = os.path.join(os.path.dirname(__file__), data_path, "segmented.gif")
        pred_gif_path = os.path.join(os.path.dirname(__file__), data_path, "pred.gif")

        utils.generate_gif(raw_single, raw_gif_path, fps=50, cmap_name="viridis")
        utils.generate_gif(segmented_single, segmented_gif_path, fps=50, cmap_name="viridis")
        utils.generate_gif(pred_single, pred_gif_path, fps=50, cmap_name="viridis")

        # Combine GIFs for comparison
        combined_gif_path = os.path.join(os.path.dirname(__file__), data_path, "combined.gif")
        utils.combine_gifs(raw_gif_path, segmented_gif_path, pred_gif_path, out_path=combined_gif_path, fps=50)
    
    print(f"Saved prediction GIFs to {data_path}")