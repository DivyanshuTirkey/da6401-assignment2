import argparse
import wandb

from utils import download_dataset
from model import NatureCNN
from data import NatureDataModule
from train import train_model
from evaluate import evaluate_best_model
from sweep import run_sweep, analyze_sweep_results

def main():
    parser = argparse.ArgumentParser(description='Train CNN on iNaturalist dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'sweep', 'analyze', 'evaluate'], 
                        default='train', help='Operation mode')
    parser.add_argument('--sweep_id', type=str, default=None, help='Sweep ID for analysis')
    parser.add_argument('--sweep_count', type=int, default=30, help='Number of sweep runs')
    parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key')
    
    args = parser.parse_args()
    
    # Login to wandb if key is provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    
    # Download dataset
    download_dataset()
    
    if args.mode == 'train':
        # Train a default model
        config = {
            'base_filters': 32,
            'filter_strategy': 'double',
            'base_kernel': 3,
            'kernel_strategy': 'same',
            'batch_norm': True,
            'conv_activation': 'relu',
            'dense_activation': 'relu',
            'dense_size': 128,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'dropout': 0.2,
            'batch_size': 64,
            'data_aug': True
        }
        model, results = train_model(config)
        print(f"Test results: {results}")
        
    elif args.mode == 'sweep':
        # Run hyperparameter sweep
        sweep_id = run_sweep(count=args.sweep_count)
        print(f"Sweep completed. Sweep ID: {sweep_id}")
        
    elif args.mode == 'analyze':
        # Analyze sweep results
        if not args.sweep_id:
            print("Error: sweep_id is required for analysis mode")
            return
        best_config = analyze_sweep_results(args.sweep_id)
        print("\nBest configuration:")
        for k, v in best_config.items():
            print(f"- {k}: {v}")
            
    elif args.mode == 'evaluate':
        # Evaluate the best model
        if not args.sweep_id:
            print("Error: sweep_id is required for evaluation mode")
            return
        best_config = analyze_sweep_results(args.sweep_id)
        model = evaluate_best_model(best_config)
        
if __name__ == "__main__":
    main()
