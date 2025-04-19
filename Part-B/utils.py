import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import subprocess

def download_dataset(url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip", output_dir="./"):
    """Download and extract the iNaturalist dataset"""
    # Check if dataset already exists
    if os.path.exists(os.path.join(output_dir, 'inaturalist_12K')):
        print("Dataset already exists. Skipping download.")
        return
    
    zip_path = os.path.join(output_dir, "nature_12K.zip")
    
    print("Downloading iNaturalist dataset...")
    subprocess.run(["curl", url, "--output", zip_path])
    
    print("Extracting dataset...")
    subprocess.run(["unzip", zip_path, "-d", output_dir])
    
    print("Cleaning up...")
    os.remove(zip_path)
    
    print("Dataset ready!")

def visualize_predictions(model, dataloader, device, class_names):
    """Create a visualization of model predictions"""
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # Plot a grid of images with predictions
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(min(32, len(images))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        pred_class_name = class_names[preds[i].item()]
        true_class_name = class_names[labels[i].item()]
        color = 'green' if preds[i] == labels[i] else 'red'
        axes[i].set_title(f"Pred: {pred_class_name}", color=color)
        axes[i].axis('off')

    plt.tight_layout()
    wandb.log({"test_predictions": wandb.Image(fig)})
    plt.close()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation loss/accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    wandb.log({"training_curves": wandb.Image(fig)})
    plt.close(fig)

def count_parameters(model):
    """Count trainable and total parameters in a model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": 100 * trainable_params / total_params
    }
