import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import wandb

from model import NatureCNN
from data import NatureDataModule
from utils import visualize_model

def train_sweep():
    """Train a model with the specified hyperparameters and log results to W&B"""
    # Initialize wandb run
    with wandb.init() as run:
        config = wandb.config
        wandb_logger = WandbLogger(log_model='all')
        
        # Log hyperparameters to be used
        print(f"Training with hyperparameters:")
        for key, value in config.items():
            print(f"- {key}: {value}")
        
        # Initialize data module
        dm = NatureDataModule(
            image_size=128,
            batch_size=config.batch_size,
            data_aug=config.data_aug
        )
        
        # Initialize model from sweep config
        model = NatureCNN(
            base_filters=config.base_filters,
            filter_strategy=config.filter_strategy,
            base_kernel=config.base_kernel,
            kernel_strategy=config.kernel_strategy,
            batch_norm=config.batch_norm,
            conv_activation=config.conv_activation,
            dense_size=config.dense_size,
            dense_activation=config.dense_activation,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
        )
        
        # Log model architecture
        wandb.log({"model_summary": str(model)})
        
        # Set up trainer with early stopping and model checkpointing
        trainer = L.Trainer(
            max_epochs=5,
            logger=wandb_logger,
            callbacks=[
                EarlyStopping(monitor='val_acc', mode='max', patience=3),
                ModelCheckpoint(monitor='val_acc', mode='max', filename='best-{epoch:02d}-{val_acc:.4f}')
            ],
            precision='16-mixed',  # Use mixed precision for faster training
            accelerator='auto',
            devices=1,
            log_every_n_steps=1000
        )
        
        # Train model
        trainer.fit(model, dm)
        
        # Test best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        
        return best_model_path

def train_model(config, max_epochs=25, min_epochs=17):
    """Train a model with the specified configuration for a longer duration"""
    # Initialize wandb
    wandb.init(project="Assignment_CNN-partA", name="best_model_training")
    wandb_logger = WandbLogger(log_model='all')
    
    # Initialize data module
    dm = NatureDataModule(
        image_size=128,
        batch_size=config['batch_size'],
        data_aug=config['data_aug']
    )
    dm.setup()
    
    # Initialize model with best configuration
    model = NatureCNN(
        base_filters=config['base_filters'],
        filter_strategy=config['filter_strategy'],
        base_kernel=config['base_kernel'],
        kernel_strategy=config['kernel_strategy'],
        batch_norm=config['batch_norm'],
        conv_activation=config['conv_activation'],
        dense_activation=config['dense_activation'],
        dense_size=config['dense_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        dropout=config['dropout']
    )
    
    # Set up trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor='val_acc', mode='max', patience=3),
            ModelCheckpoint(monitor='val_acc', mode='max', filename='best-{epoch:02d}-{val_acc:.4f}')
        ],
        precision='16-mixed',  # Use mixed precision for faster training
        accelerator='auto',
        devices=1,
        log_every_n_steps=1000
    )
    
    # Train and test the model
    trainer.fit(model, dm)
    test_results = trainer.test(model, dm)
    
    # Create and log visualization of predictions
    visualize_model(model, dm)
    
    wandb.finish()
    
    return model, test_results
