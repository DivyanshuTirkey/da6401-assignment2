import argparse
import torch
import wandb
import os

from model import PretrainedModel
from data import NatureDataModule
from train import train_model
from evaluate import evaluate_model
from utils import download_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained models on iNaturalist dataset")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="resnet50", 
                        choices=["resnet50", "vgg16", "efficientnet_b0"],
                        help="Pre-trained model architecture")
    parser.add_argument("--freeze_strategy", type=str, default="feature_extraction",
                        choices=["feature_extraction", "partial_finetuning", "progressive_unfreezing"],
                        help="Layer freezing strategy")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="./inaturalist_12K", help="Data directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--data_aug", action="store_true", help="Use data augmentation")
    
    # Run configuration
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                        help="Run mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint to load")
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")
    
    args = parser.parse_args()
    
    # Login to wandb if key is provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    
    # Download dataset if needed
    if not os.path.exists(args.data_dir):
        download_dataset(output_dir=os.path.dirname(args.data_dir))
    
    # Create configuration dictionary
    config = {
        "model_name": args.model_name,
        "freeze_strategy": args.freeze_strategy,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "image_size": args.image_size,
        "data_aug": args.data_aug
    }
    
    # Initialize data module
    data_module = NatureDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        data_aug=args.data_aug
    )
    data_module.setup()
    
    # Initialize model
    model = PretrainedModel(
        model_name=args.model_name,
        num_classes=len(data_module.classes),
        dropout=args.dropout,
        freeze_strategy=args.freeze_strategy
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    
    if args.mode == "train":
        # Train the model
        model, accuracy = train_model(model, data_module, config)
        print(f"Training completed with best validation accuracy: {accuracy:.4f}")
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, data_module, config)
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
    elif args.mode == "evaluate":
        # Evaluate the model
        if args.checkpoint is None:
            print("Warning: No checkpoint provided for evaluation. Using randomly initialized weights.")
        
        accuracy = evaluate_model(model, data_module, config)
        print(f"Test accuracy: {accuracy:.4f}")
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
