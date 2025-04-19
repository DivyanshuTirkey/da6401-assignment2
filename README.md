# CNN Models for iNaturalist Dataset Classification

### WandB Report: https://wandb.ai/da24m005-iit-madras/Assignment_CNN-partA/reports/DA6401-Assignment-2--VmlldzoxMjM2ODA0MA?accessToken=pomnmm5c0836vjou2q9v2tpmauuyhb1cueojq4svsl6yv3d32m0yg9doidwp9p4k

This repository contains two main components:
- **Part-A**: A custom CNN architecture trained from scratch on the iNaturalist dataset
- **Part-B**: Fine-tuning pre-trained models from torchvision on the iNaturalist dataset

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Part A: Custom CNN](#part-a-custom-cnn)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluation](#evaluation)
  - [Configuration Options](#configuration-options-part-a)
- [Part B: Pre-trained Models](#part-b-pre-trained-models)
  - [Available Models](#available-models)
  - [Fine-tuning Strategies](#fine-tuning-strategies)
  - [Training](#training-1)
  - [Evaluation](#evaluation-1)
  - [Configuration Options](#configuration-options-part-b)
- [Results](#results)
- [Project Structure](#project-structure)
- [Customization](#customization)

## Installation

```bash
# Clone the repository
git clone https://github.com/DivyanshuTirkey/da6401-assignment2.git
cd da6401-assignment2

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
lightning>=2.0.0
wandb
numpy
matplotlib
```

## Dataset

The iNaturalist dataset used in this project contains images of 10 different natural categories. The dataset will be automatically downloaded when running the scripts.

```bash
# To manually download the dataset
python -c "from Part-A.utils import download_dataset; download_dataset()"
```

## Part A: Custom CNN

### Model Architecture

The NatureCNN model is a flexible CNN architecture with 5 convolutional blocks, each followed by activation and max-pooling. The model is highly configurable with options for:

- Number of filters in each layer
- Filter organization strategy (double, halve, same, alternate)
- Kernel sizes and strategies
- Activation functions
- Batch normalization
- Dense layer size

### Training

```bash
# Train with default configuration
python -m Part-A.main --mode train

# Train with specific configuration
python -m Part-A.main --mode train --base_filters 32 --filter_strategy double --batch_norm True
```

### Hyperparameter Tuning

```bash
# Run hyperparameter sweep
python -m Part-A.main --mode sweep --sweep_count 30

# Analyze sweep results
python -m Part-A.main --mode analyze --sweep_id 

# Evaluate best model from sweep
python -m Part-A.main --mode evaluate --sweep_id 
```

### Evaluation

```bash
# Evaluate a trained model
python -m Part-A.main --mode evaluate --model_path 
```

### Configuration Options (Part A)

The NatureCNN model can be configured with the following parameters:

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `base_filters` | Initial number of filters | 32 | Integer values (e.g., 16, 32, 64) |
| `filter_strategy` | How filters scale across layers | 'double' | 'double', 'halve', 'alternate', 'same' |
| `base_kernel` | Base kernel size | 3 | Odd integers (e.g., 3, 5, 7) |
| `kernel_strategy` | How kernel sizes vary | 'same' | 'same', 'decrease', 'alternate', 'pyramid' |
| `batch_norm` | Use batch normalization | True | True, False |
| `conv_activation` | Activation for conv layers | 'relu' | 'relu', 'gelu', 'silu', 'mish', etc. |
| `dense_activation` | Activation for dense layer | 'relu' | 'relu', 'gelu', 'silu', 'mish', etc. |
| `dense_size` | Number of neurons in dense layer | 128 | Integer values (e.g., 128, 256, 512) |
| `learning_rate` | Learning rate | 1e-3 | Float values |
| `weight_decay` | Weight decay | 1e-4 | Float values |
| `dropout` | Dropout rate | 0.2 | Float values between 0 and 1 |
| `batch_size` | Batch size | 32 | Integer values |
| `data_aug` | Use data augmentation | True | True, False |

To modify default configurations, edit the `main.py` file in the Part-A directory:

```python
# Default configuration in Part-A/main.py
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
```

## Part B: Pre-trained Models

### Available Models

The following pre-trained models from torchvision are supported:
- ResNet50
- VGG16
- EfficientNet-B0

### Fine-tuning Strategies

Three fine-tuning strategies are implemented:
1. **Feature Extraction**: Freeze all layers except the final classifier
2. **Partial Fine-tuning**: Freeze the first k layers, fine-tune the rest
3. **Progressive Unfreezing**: Start with all layers frozen except the classifier, then gradually unfreeze deeper layers

### Training

```bash
# Train with default configuration (ResNet50 with feature extraction)
python -m Part-B.main --mode train

# Train with specific configuration
python -m Part-B.main --mode train --model_name resnet50 --freeze_strategy feature_extraction --num_epochs 25
```

### Evaluation

```bash
# Evaluate a trained model
python -m Part-B.main --mode evaluate --checkpoint 
```

### Configuration Options (Part B)

The pre-trained models can be configured with the following parameters:

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model_name` | Pre-trained model architecture | 'resnet50' | 'resnet50', 'vgg16', 'efficientnet_b0' |
| `freeze_strategy` | Layer freezing strategy | 'feature_extraction' | 'feature_extraction', 'partial_finetuning', 'progressive_unfreezing' |
| `num_epochs` | Number of training epochs | 25 | Integer values |
| `batch_size` | Batch size | 32 | Integer values |
| `learning_rate` | Learning rate | 1e-4 | Float values |
| `dropout` | Dropout rate | 0.2 | Float values between 0 and 1 |
| `image_size` | Input image size | 224 | Integer values |
| `data_aug` | Use data augmentation | True | True, False |

To modify default configurations, edit the `main.py` file in the Part-B directory:

```python
# Default configuration in Part-B/main.py
config = {
    "model_name": "resnet50",
    "freeze_strategy": "feature_extraction",
    "num_epochs": 25,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "dropout": 0.2,
    "image_size": 224,
    "data_aug": True
}
```

## Results

### Part A: Custom CNN

The best model configuration from our hyperparameter sweep achieved 37.3% validation accuracy with the following parameters:
- base_filters: 32
- filter_strategy: double
- batch_norm: True
- conv_activation: gelu

### Part B: Pre-trained Models

The ResNet50 model with feature extraction (freezing all layers except the last) achieved 78% test accuracy after 25 epochs of training, with only 0.09% of the parameters being trainable.

## Project Structure

```
iNaturalist-CNN-Project/
├── README.md
├── requirements.txt
├── partA.ipynb
├── partB.ipynb
├── Part-A/
│   ├── __init__.py
│   ├── main.py         # Main script for Part A
│   ├── model.py        # NatureCNN model definition
│   ├── data.py         # Data loading and processing
│   ├── utils.py        # Utility functions
│   ├── train.py        # Training functions
│   ├── evaluate.py     # Evaluation functions
│   └── sweep.py        # Hyperparameter tuning
└── Part-B/
    ├── __init__.py
    ├── main.py         # Main script for Part B
    ├── model.py        # Pre-trained model setup
    ├── data.py         # Data loading and processing
    ├── utils.py        # Utility functions
    ├── train.py        # Training functions
    └── evaluate.py     # Evaluation functions
```

## Customization

### Adding a New Activation Function

To add a new activation function, modify the `_get_activation` method in `Part-A/model.py`:

```python
def _get_activation(self, name: str) -> nn.Module:
    """Map activation name to PyTorch module"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'your_new_activation': YourNewActivation()  # Add your new activation here
    }
    return activations[name.lower()]
```

### Adding a New Pre-trained Model

To add a new pre-trained model, modify the `_load_pretrained_model` method in `Part-B/model.py`:

```python
def _load_pretrained_model(self, model_name, num_classes, dropout):
    """Load a pre-trained model and modify the classifier for our dataset"""
    if model_name == 'resnet50':
        # ResNet50 implementation
    elif model_name == 'vgg16':
        # VGG16 implementation
    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0 implementation
    elif model_name == 'your_new_model':
        # Your new model implementation
        model = models.your_new_model(pretrained=True)
        # Modify the classifier
        # ...
    else:
        raise ValueError(f"Unsupported model: {model_name}")
            
    return model
```

### Adding a New Freezing Strategy

To add a new freezing strategy, modify the `apply_freeze_strategy` method in `Part-B/model.py`:

```python
def apply_freeze_strategy(self, strategy, current_epoch=None):
    """Apply the specified freezing strategy to the model"""
    if strategy == 'feature_extraction':
        # Feature extraction implementation
    elif strategy == 'partial_finetuning':
        # Partial fine-tuning implementation
    elif strategy == 'progressive_unfreezing':
        # Progressive unfreezing implementation
    elif strategy == 'your_new_strategy':
        # Your new strategy implementation
        # ...
```