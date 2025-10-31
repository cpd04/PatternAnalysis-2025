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
from dataset import load_prostate_data
from modules import ImprovedUNet

# Parameters for Training
BATCH_SIZE=2
EPOCHS=20
LEARNING_RATE=4e-4
MODEL_PATH = os.path.join("./models/", "3d_unet_model.pth")
VISUAL_PATH_OUTPUTS = os.path.join("./visualisation/outputs/")

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
        outputs = model(raw.to(device))              # (N, C, H, W)

    # Convert to single channel for visualisation
    segmented_single = utils.to_single_channel(segmented[0]).cpu().numpy()
    pred_single = utils.to_single_channel(outputs[0]).cpu().numpy()
    raw_single = raw[0].cpu().numpy()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), data_path), exist_ok=True)

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
    return [raw_gif_path, segmented_gif_path, pred_gif_path, combined_gif_path]

def load_and_predict_model(model_path: str, data_path: str, visual_path: str, device: str = "cuda"):
    """
    Load a trained model and perform prediction on a single image.

    Args:
        model_path: path to the saved model file
        data_path: path to save visualisations
        loader: DataLoader for the test image
        device: device string ("cuda" or "cpu")
    """
    # Load the trained model
    model = ImprovedUNet(in_channels=1, out_channels=6).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    print(f"Loaded model from {model_path}")

    # Set model to evaluation mode
    model.eval()

    # Import test data loader
    _, _, test_loader = load_prostate_data(data_path, 
                                            train_data=0, 
                                            validation_data=0, 
                                            test_data=1, 
                                            train_split=0.7, 
                                            validation_split=0.15,
                                            batch_size=BATCH_SIZE,
                                            downsample=True,
                                            debugging_mode=False)

    # Perform prediction
    evaluate_unet(model, test_loader, device=device)
    predict_single_image(model, test_loader, visual_path, device=device)

if __name__ == "__main__":
    # Define paths
    data_path = os.path.join(os.path.dirname(__file__), "data", "HipMRI_study_complete_release_v1")
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and perform prediction
    load_and_predict_model(model_path, data_path, VISUAL_PATH_OUTPUTS, device=device)