import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

from .model import NatureCNN
from .data import NatureDataModule
from .utils import visualize_test_predictions, visualize_filters

def evaluate_best_model(best_config):
    """Evaluate the best model from the sweep on the test set"""
    # Initialize wandb
    wandb.init(project="Assignment_CNN-partA", name="best_model_evaluation")
    wandb_logger = WandbLogger(log_model='all')
    
    # Log the best configuration
    print("Best model configuration:")
    for k, v in best_config.items():
        print(f"- {k}: {v}")
    
    # Initialize data module
    dm = NatureDataModule(
        image_size=128,
        batch_size=best_config['batch_size'],
        data_aug=False  # No augmentation for evaluation
    )
    dm.setup()
    
    # Initialize model with best configuration
    model = NatureCNN(
        base_filters=best_config['base_filters'],
        filter_strategy=best_config['filter_strategy'],
        base_kernel=best_config['base_kernel'],
        kernel_strategy=best_config['kernel_strategy'],
        batch_norm=best_config['batch_norm'],
        conv_activation=best_config['conv_activation'],
        dense_activation=best_config['dense_activation'],
        dense_size=best_config['dense_size'],
        learning_rate=best_config['learning_rate'],
        weight_decay=best_config['weight_decay'],
        dropout=best_config['dropout']
    )
    
    # Set up trainer
    trainer = L.Trainer(
        max_epochs=25,
        min_epochs=17,
        logger=wandb_logger,
        callbacks=[],
        precision='16-mixed',  # Use mixed precision for faster training
        accelerator='auto',
        devices=1,
        log_every_n_steps=1000
    )
    
    # Test the model
    trainer.test(model, dm)
    
    # Create and log visualization of predictions
    visualize_test_predictions(model, dm)
    
    # Visualize filters
    visualize_filters(model)
    
    wandb.finish()
    
    return model
