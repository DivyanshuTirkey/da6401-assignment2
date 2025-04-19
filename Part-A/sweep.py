import wandb
import pandas as pd
import numpy as np
from .train import train_sweep

# WandB sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'base_filters': {
            'values': [16, 32, 64]
        },
        'filter_strategy': {
            'values': ['double', 'halve', 'alternate', 'same']
        },
        'base_kernel': {
            'values': [3, 5, 7]
        },
        'kernel_strategy': {
            'values': ['same', 'decrease', 'alternate', 'pyramid']
        },
        'batch_norm': {
            'values': [True, False]
        },
        'conv_activation': {
            'values': ['relu', 'gelu', 'silu', 'mish']
        },
        'dense_activation': {
            'values': ['relu', 'gelu', 'silu', 'mish']
        },
        'dense_size': {
            'values': [128, 256, 512]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'dropout': {
            'values': [0.0, 0.2, 0.3, 0.4, 0.5]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'data_aug': {
            'values': [True, False]
        }
    }
}

def analyze_sweep_results(sweep_id):
    """Analyze the results from the hyperparameter sweep and generate insights"""
    # Initialize wandb API
    api = wandb.Api()
    
    # Get the sweep runs
    sweep = api.sweep(f"da24m005-iit-madras/Assignment_CNN-partA/sweeps/{sweep_id}")
    runs = sorted(sweep.runs, key=lambda run: run.summary.get('val_acc', 0), reverse=True)
    
    print(f"Total runs: {len(runs)}")
    print(f"Best validation accuracy: {runs[0].summary.get('val_acc', 0):.4f}")
    
    # Extract configurations and metrics for all runs
    configs = []
    metrics = []
    
    for run in runs:
        config = {k: v for k, v in run.config.items() 
                 if not k.startswith('_') and k != 'wandb'}
        metric = {'val_acc': run.summary.get('val_acc', 0),
                 'test_acc': run.summary.get('test/acc', 0)}
        configs.append(config)
        metrics.append(metric)
    
    # Create a DataFrame
    df = pd.DataFrame([{**c, **m} for c, m in zip(configs, metrics)])
    
    # Handle non-numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation with validation accuracy
    corr = numeric_df.corr()['val_acc'].sort_values(ascending=False)
    print("\nCorrelation with validation accuracy:")
    print(corr)
    
    # Analyze effect of filter strategy
    print("\nEffect of filter strategy:")
    filter_strategy_effect = df.groupby('filter_strategy')['val_acc'].agg(['mean', 'max', 'count'])
    print(filter_strategy_effect.sort_values('max', ascending=False))
    
    # Analyze effect of activation function
    print("\nEffect of activation function:")
    activation_effect = df.groupby('conv_activation')['val_acc'].agg(['mean', 'max', 'count'])
    print(activation_effect.sort_values('max', ascending=False))
    
    # Analyze effect of batch normalization
    print("\nEffect of batch normalization:")
    bn_effect = df.groupby('batch_norm')['val_acc'].agg(['mean', 'max', 'count'])
    print(bn_effect)
    
    # Analyze effect of data augmentation
    print("\nEffect of data augmentation:")
    aug_effect = df.groupby('data_aug')['val_acc'].agg(['mean', 'max', 'count'])
    print(aug_effect)
    
    # Generate insights
    print("\nKey insights from hyperparameter sweep:")
    
    # Insight 1: Filter strategy
    best_filter = filter_strategy_effect.index[filter_strategy_effect['max'].argmax()]
    print(f"1. Filter strategy: '{best_filter}' performed best, suggesting that "
          f"{'increasing filter complexity in deeper layers' if best_filter == 'double' else 'maintaining consistent filters across layers' if best_filter == 'same' else 'using alternating filter patterns' if best_filter == 'alternate' else 'reducing filter complexity in deeper layers'} "
          f"is effective for this dataset.")
    
    # Insight 2: Activation function
    best_activation = activation_effect.index[activation_effect['max'].argmax()]
    print(f"2. Activation function: '{best_activation}' yielded the highest accuracy, "
          f"which may be due to its {'better gradient flow' if best_activation in ['gelu', 'silu', 'mish'] else 'simplicity and efficiency' if best_activation == 'relu' else 'special properties'}.")
    
    # Insight 3: Batch normalization
    bn_better = bn_effect.loc[True, 'mean'] > bn_effect.loc[False, 'mean']
    print(f"3. Batch normalization {'improved' if bn_better else 'did not significantly improve'} model performance, "
          f"suggesting it {'helps normalize feature distributions' if bn_better else 'may not be necessary for this dataset'}.")
    
    # Insight 4: Data augmentation
    aug_better = aug_effect.loc[True, 'mean'] > aug_effect.loc[False, 'mean']
    print(f"4. Data augmentation {'improved' if aug_better else 'did not significantly improve'} generalization, "
          f"indicating that {'increasing dataset diversity helps prevent overfitting' if aug_better else 'the dataset may already contain sufficient variety'}.")
    
    # Return the best configuration
    best_config = {k: runs[0].config[k] for k in configs[0].keys()}
    return best_config

def run_sweep(count=30):
    """Start a new sweep and run the specified number of trials"""
    sweep_id = wandb.sweep(sweep_config, project="Assignment_CNN-partA")
    wandb.agent(sweep_id, train_sweep, count=count)
    return sweep_id
