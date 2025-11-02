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
# Ensure that your environment has WANDB has api key

# Import required libraries
import torch
import torch.nn as nn

import os
import wandb

from modules import ImprovedUNet
from dataset import load_prostate_data
import utils
import predict

# Parameters for Training
BATCH_SIZE=2
EPOCHS=30
LEARNING_RATE=4e-4
LEARNING_RATE_MIN=1e-4
MODEL_PATH = os.path.join("./models/")
VISUAL_PATH_INPUTS = os.path.join("./visualisation/inputs/")
VISUAL_PATH_VALIDATION = os.path.join("./visualisation/validation/")
VISUAL_PATH_OUTPUTS = os.path.join("./visualisation/outputs/")

def visualise_inputs(loader, output_path):
    """
    Visualise first sample inputs from the dataloader and save as gifs.
    
    Arguments:
        loader: DataLoader to get samples from
        output_path: path to save visualisations
    
    Returns:
        sample_raw_path: path to saved raw input gif
        sample_segmented_path: path to saved segmented input gif
    """
    
    # Get a batch of data
    for raw, segmented in loader:
        break

    # Visualise the first sample in the batch
    sample_raw = raw[0].cpu().numpy()  # Shape: (1, D, H, W)
    sample_segmented = segmented[0]    # Shape: (C, D, H, W)
    segmented_converted = utils.to_single_channel(sample_segmented).cpu().numpy()  # Shape: (1, D, H, W)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), output_path), exist_ok=True)

    # Save visualisations
    sample_raw_path = os.path.join(os.path.dirname(__file__), output_path, "sample_raw.gif")
    sample_segmented_path = os.path.join(os.path.dirname(__file__), output_path, "sample_segmented.gif")

    # Generate GIFs
    utils.generate_gif(sample_raw, sample_raw_path, fps=50, cmap_name="viridis")
    utils.generate_gif(segmented_converted, sample_segmented_path, fps=50, cmap_name="viridis")

    return [sample_raw_path, sample_segmented_path]

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation. Used as a criterion for 
    training the UNet model.
    """
    def __init__(self, smooth=1e-6):
        """
        Arguments:
            smooth: small constant to avoid division by zero
        """
        
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, probs, targets):
        """
        Compute the Soft Dice Loss.

        Arguments:
            probs: [B, C, D, H, W] raw network outputs (softmax applied)
            targets: [B, C, D, H, W] one-hot ground truth
        
        Returns:
            dice_loss: computed Soft Dice Loss
        """
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
    Load the data and train the model to perform predictions. Visualisations
    are created and saved for analysis.

    Returns:
        model : trained UNet model
    """
    # Start a new run
    wandb.init(project="COMP3710-training",
                name="transformed_improved_3D_Unet_full_data",
                config={
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "optimizer": "adam"
                })

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    data_path = os.path.join(os.path.dirname(__file__), "data", "HipMRI_study_complete_release_v1")
    train_loader, validation_loader, test_loader = load_prostate_data(data_path, 
                                                            train_data=True, 
                                                            validation_data=True, 
                                                            test_data=True, 
                                                            train_split=0.7, 
                                                            validation_split=0.15,
                                                            batch_size=BATCH_SIZE,
                                                            downsample=False,
                                                            transform_flag=True,
                                                            debugging_mode=False)
    
    # Visualise some inputs and add to W&B
    sample_raw_path, sample_segmented_path = visualise_inputs(train_loader, VISUAL_PATH_INPUTS)
    wandb.log({
        "train_visual/sample_raw": wandb.Video(sample_raw_path, format="gif"),
        "train_visual/sample_segmented": wandb.Video(sample_segmented_path, format="gif")
    })
    
    # Initialise the model, loss function, and optimizer
    model = ImprovedUNet(in_channels=1, out_channels=6).to(device, dtype=torch.float32)
    criterion = SoftDiceLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE_MIN)

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
            raw = raw.to(device, dtype=torch.float32)
            segmented = segmented.to(device, dtype=torch.uint8)

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
                    val_raw = val_raw.to(device, dtype=torch.float32)
                    val_segmented = val_segmented.to(device, dtype=torch.uint8)

                    val_outputs = model(val_raw)
                    v_loss = criterion(val_outputs, val_segmented)
                    val_loss += v_loss.item() * val_raw.size(0)

            val_loss /= len(validation_loader.dataset)

            # Check DICE scores on validation set
            dice_scores, multi_dsc = predict.evaluate_unet(model, validation_loader, device=device)

            # Log DICE scores to W&B
            for cls, dsc in enumerate(dice_scores):
                wandb.log({f"validation/dice_score/class_{cls}": dsc, "epoch": epoch+1})
            wandb.log({"validation/dice_score/multiclass": multi_dsc, "epoch": epoch+1})

        if val_loss is not None:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}")

        # Print DICE scores
        print(f"Multiclass Dice Score on Validation Set: {multi_dsc:.4f}")

        # Log to W&B
        wandb.log({
            "training/loss": avg_loss,
            "validation/loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch+1,
        })

        # Visualise predictions on a sample from validation set
        if validation_loader is not None:
            # Generate gifs during training
            val_path = os.path.join(os.path.dirname(__file__), VISUAL_PATH_VALIDATION)
            os.makedirs(os.path.dirname(val_path), exist_ok=True)
            vis_epoch = os.path.join(val_path, f"epoch_{epoch+1}")
            raw_gif_path, segmented_gif_path, pred_gif_path, combined_gif_path = predict.predict_single_image(model, validation_loader, vis_epoch, device=device)
            
            # Upload visualisation gif to W&B
            wandb.log({
                f"val_visual/combined/epoch_{epoch+1}": wandb.Video(combined_gif_path, format="gif"),
            })

        # Step the scheduler
        scheduler.step()
    
    # Save the trained model
    model_path_dir = os.path.join(os.path.dirname(__file__), MODEL_PATH)

    # Create directory if it doesn't exist and save
    os.makedirs(os.path.dirname(model_path_dir), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path_dir, "3d_unet_model.pth"))
    print(f"Trained model saved at {model_path_dir}")

    # Evaluate on validation set
    dice_class, multi_dsc = predict.evaluate_unet(model, test_loader, device=device)
    print("Average Dice Score per class on Test Set:")
    
    for cls, dsc in enumerate(dice_class):
        print(f"  Class {cls}: {dsc:.4f}")
        wandb.log({f"test/dice_score/class_{cls}": dsc})

    print(f"Multiclass Dice Score on Test Set: {multi_dsc:.4f}")
    wandb.log({"test/dice_score/multiclass": multi_dsc})
    
    # Visualise predictions on a sample image
    output_paths = os.path.join(os.path.dirname(__file__), VISUAL_PATH_OUTPUTS)
    raw_gif_path, segmented_gif_path, pred_gif_path, combined_gif_path = predict.predict_single_image(model, test_loader, output_paths, device=device)

    # Upload visualisation gif to W&B
    wandb.log({"test_visual/combined": wandb.Video(combined_gif_path, format="gif"),
               "test_visual/raw": wandb.Video(raw_gif_path, format="gif"),
               "test_visual/segmented": wandb.Video(segmented_gif_path, format="gif"),
               "test_visual/predicted": wandb.Video(pred_gif_path, format="gif")
               })

    return model

if __name__ == "__main__":
    trained_model = train_model()