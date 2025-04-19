import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
from utils import visualize_predictions, plot_training_curves, count_parameters

def train_model(model, data_module, config):
    """Train a model with the specified configuration"""
    # Initialize wandb
    wandb.init(project="Assignment2_CNN_partB", name=f"{config['model_name']}-{config['freeze_strategy']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Log parameter counts
    param_counts = count_parameters(model)
    print(f"Trainable parameters: {param_counts['trainable_parameters']:,} ({param_counts['trainable_percentage']:.2f}% of total)")
    wandb.log(param_counts)
    
    # Set up optimizer and loss function
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Get data loaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    test_loader = data_module.get_test_dataloader()
    
    # Training loop
    num_epochs = config['num_epochs']
    best_accuracy = 0.0
    best_model_path = os.path.join(wandb.run.dir, "best_model.pth")
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # For progressive unfreezing, update which layers are frozen
        if config['freeze_strategy'] == 'progressive_unfreezing':
            model.apply_freeze_strategy('progressive_unfreezing', current_epoch=epoch)
            # Update optimizer to include newly unfrozen parameters if needed
            if epoch == 5:  # Unfreeze epoch
                optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'] * 0.5)
                print("Unfreezing layer4, adjusting learning rate")
                # Update parameter counts after unfreezing
                new_param_counts = count_parameters(model)
                print(f"Updated trainable parameters: {new_param_counts['trainable_parameters']:,} ({new_param_counts['trainable_percentage']:.2f}% of total)")
                wandb.log(new_param_counts)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    return model, best_accuracy
