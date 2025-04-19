import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

def visualize_model(model, dm):
    """Create visualizations for model analysis and log to W&B"""
    # Get a batch of test data
    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))
    images, labels = batch
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
    
    # Log sample predictions
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            # Convert tensor to numpy image
            img = images[i].permute(1, 2, 0).cpu().numpy()
            # Denormalize the image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            true_class = dm.classes[labels[i]]
            pred_class = dm.classes[preds[i]]
            color = 'green' if preds[i] == labels[i] else 'red'
            ax.set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
            ax.axis('off')
    
    plt.tight_layout()
    wandb.log({"prediction_samples": wandb.Image(fig)})
    plt.close(fig)
    
    # Visualize first layer filters
    if hasattr(model.features[0], 'weight'):
        filters = model.features[0].weight.detach().cpu()
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        for i, ax in enumerate(axes.flatten()):
            if i < filters.shape[0]:
                # Take mean across input channels
                ax.imshow(filters[i].mean(0), cmap='viridis')
                ax.axis('off')
        plt.tight_layout()
        wandb.log({"first_layer_filters": wandb.Image(fig)})
        plt.close(fig)
    
    # Log confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        images, labels = batch
        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Create confusion matrix
    confusion = wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=dm.classes
    )
    wandb.log({"confusion_matrix": confusion})

def visualize_test_predictions(model, dm):
    """Create a grid of test images with model predictions"""
    # Get test dataloader
    test_loader = dm.test_dataloader()
    
    # Get a batch of test images
    all_images = []
    all_labels = []
    all_preds = []
    
    # Get 30 random samples for visualization
    for images, labels in test_loader:
        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
        
        all_images.extend(images.cpu())
        all_labels.extend(labels.cpu())
        all_preds.extend(preds.cpu())
        
        if len(all_images) >= 30:
            break
    
    # Convert to numpy arrays
    all_images = [img.permute(1, 2, 0).numpy() for img in all_images[:30]]
    all_labels = [label.item() for label in all_labels[:30]]
    all_preds = [pred.item() for pred in all_preds[:30]]
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    all_images = [np.clip(std * img + mean, 0, 1) for img in all_images]
    
    # Create a grid of images
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    
    for i, (img, label, pred) in enumerate(zip(all_images, all_labels, all_preds)):
        row = i % 10
        col = i // 10
        
        ax = axes[row, col]
        ax.imshow(img)
        
        true_class = dm.classes[label]
        pred_class = dm.classes[pred]
        
        # Green for correct, red for incorrect
        color = 'green' if label == pred else 'red'
        
        ax.set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    wandb.log({"test_predictions_grid": wandb.Image(fig)})
    plt.close(fig)

def visualize_filters(model):
    """Visualize filters in the first convolutional layer"""
    # Get the first convolutional layer
    first_conv = model.features[0]
    
    # Get the weights
    weights = first_conv.weight.detach().cpu()
    
    # Create a grid of filter visualizations
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flatten()):
        if i < weights.shape[0]:
            # Normalize the filter for visualization
            filter_img = weights[i].mean(0)  # Average across input channels
            filter_min, filter_max = filter_img.min(), filter_img.max()
            filter_img = (filter_img - filter_min) / (filter_max - filter_min + 1e-8)
            
            ax.imshow(filter_img, cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    wandb.log({"first_layer_filters": wandb.Image(fig)})
    plt.close(fig)

def download_dataset():
    """Download and extract the iNaturalist dataset"""
    import os
    import subprocess
    
    # Check if dataset already exists
    if os.path.exists('/kaggle/working/inaturalist_12K'):
        print("Dataset already exists. Skipping download.")
        return
    
    print("Downloading iNaturalist dataset...")
    subprocess.run(["curl", "https://storage.googleapis.com/wandb_datasets/nature_12K.zip", "--output", "nature_12K.zip"])
    
    print("Extracting dataset...")
    subprocess.run(["unzip", "nature_12K.zip", "-d", "/kaggle/working/"])
    
    print("Cleaning up...")
    subprocess.run(["rm", "nature_12K.zip"])
    
    print("Dataset ready!")
