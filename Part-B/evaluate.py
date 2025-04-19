import torch
import wandb
from tqdm import tqdm
from .utils import visualize_predictions

def evaluate_model(model, data_module, config):
    """Evaluate a model on the test set"""
    # Initialize wandb if not already initialized
    if wandb.run is None:
        wandb.init(project="Assignment2_CNN_partB", name=f"eval-{config['model_name']}-{config['freeze_strategy']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Get test dataloader
    test_loader = data_module.get_test_dataloader()
    
    # Evaluation
    correct = 0
    total = 0
    class_correct = [0] * len(data_module.classes)
    class_total = [0] * len(data_module.classes)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on test set"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate metrics
    accuracy = correct / total
    
    # Log overall accuracy
    print(f"Test Accuracy: {accuracy:.4f}")
    wandb.log({"test_accuracy": accuracy})
    
    # Log per-class accuracy
    class_accuracies = {}
    for i in range(len(data_module.classes)):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            class_accuracies[f"class_acc/{data_module.classes[i]}"] = class_acc
            print(f"Accuracy of {data_module.classes[i]}: {class_acc:.4f}")
    
    wandb.log(class_accuracies)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, device, data_module.classes)
    
    return accuracy
