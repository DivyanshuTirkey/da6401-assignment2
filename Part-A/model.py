import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy

class NatureCNN(L.LightningModule):
    def __init__(self,
                base_filters=32,
                filter_strategy='double',
                base_kernel=3,
                kernel_strategy='same',
                batch_norm=True,
                conv_activation='relu',
                dense_activation='relu',
                dense_size=128,
                learning_rate=1e-3,
                weight_decay=1e-4,
                dropout=0.2):
        """
        Flexible CNN model for iNaturalist dataset classification.
        
        Args:
            base_filters: Starting number of filters in first convolutional layer
            filter_strategy: Strategy for scaling filters across layers ['same', 'double', 'halve', 'alternate']
            base_kernel: Base kernel size for convolutional layers
            kernel_strategy: Strategy for kernel sizes ['same', 'decrease', 'alternate', 'pyramid']
            batch_norm: Whether to use batch normalization
            conv_activation: Activation function for convolutional layers
            dense_activation: Activation function for dense layers
            dense_size: Number of neurons in dense layer
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            dropout: Dropout rate
        """
        super().__init__()
        self.save_hyperparameters()

        # Generate architecture configuration
        self.filters = self.get_filter_strategy(base_filters, filter_strategy)
        self.kernel_sizes = self.generate_kernel_sizes(base_kernel, kernel_strategy)
        
        # Validate configuration
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError("Filter numbers and kernel sizes must match")

        # Build convolutional blocks
        self.features = nn.Sequential()
        in_channels = 3
        
        for i, (out_channels, k_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            # Ensure odd kernel size for symmetric padding
            k_size = k_size if k_size % 2 else k_size + 1
            padding = k_size // 2
            
            # Add logging for model construction
            print(f"Layer {i+1}: {in_channels} -> {out_channels}, kernel: {k_size}x{k_size}")
            
            self.features.append(nn.Conv2d(in_channels, out_channels, k_size, padding=padding))
            
            if self.hparams.batch_norm:
                self.features.append(nn.BatchNorm2d(out_channels))
            
            self.features.append(self._get_activation(self.hparams.conv_activation))
            self.features.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        # Calculate classifier input size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            self.feature_size = self.features(dummy).flatten(1).size(1)

        # Build classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, dense_size),
            self._get_activation(self.hparams.dense_activation),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 10)  # 10 classes in the iNaturalist subset
        )

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)

    @staticmethod
    def get_filter_strategy(base: int, strategy: str, num_layers=5) -> list:
        """Generate filter numbers based on base and strategy"""
        strategies = {
            'same': [base] * num_layers,
            'double': [base * (2**i) for i in range(num_layers)],
            'halve': [max(8, base // (2**i)) for i in range(num_layers)],
            'alternate': [base * (2 if i%2 else 1) for i in range(num_layers)]
        }
        return strategies[strategy.lower()]

    @staticmethod
    def generate_kernel_sizes(base: int, strategy: str, num_layers=5) -> list:
        """Generate kernel sizes based on base and strategy"""
        strategies = {
            'same': [base] * num_layers,
            'decrease': [max(3, base - 2*i) for i in range(num_layers)],
            'alternate': [base if i%2 else (base-2) for i in range(num_layers)],
            'pyramid': [base + 2*i for i in range(num_layers//2)] + 
                      [base + 2*(num_layers//2 - i) for i in range(1, num_layers-num_layers//2+1)]
        }
        return strategies[strategy.lower()]

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
            'sigmoid': nn.Sigmoid()
        }
        return activations[name.lower()]

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def calculate_computations(self, image_size=128):
        """
        Calculate the total number of computations (FLOPs) performed by the network
        
        Args:
            image_size: Size of input image (assumed square)
            
        Returns:
            Total number of multiplication and addition operations
        """
        total_flops = 0
        input_size = image_size
        
        # Compute FLOPs for each convolutional layer
        in_channels = 3
        for i, (out_channels, k_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            # Each output pixel requires k*k*in_channels multiplications and additions
            # Number of output pixels = input_size^2 (due to padding)
            # Number of output channels = out_channels
            flops_per_layer = input_size**2 * out_channels * k_size**2 * in_channels
            
            # Add batch norm operations if used (4 operations per element)
            if self.hparams.batch_norm:
                flops_per_layer += 4 * input_size**2 * out_channels
            
            # Add activation function operations (1 operation per element)
            flops_per_layer += input_size**2 * out_channels
            
            print(f"Layer {i+1}: {flops_per_layer} FLOPs")
            total_flops += flops_per_layer
            
            # Update for next layer (maxpool reduces spatial dimensions by half)
            input_size = input_size // 2
            in_channels = out_channels
        
        # Compute FLOPs for the dense layer
        flops_dense = self.feature_size * self.hparams.dense_size
        # Add activation
        flops_dense += self.hparams.dense_size
        # Add dropout (1 operation per element)
        flops_dense += self.hparams.dense_size
        
        print(f"Dense layer: {flops_dense} FLOPs")
        total_flops += flops_dense
        
        # Compute FLOPs for the output layer
        flops_output = self.hparams.dense_size * 10
        print(f"Output layer: {flops_output} FLOPs")
        total_flops += flops_output
        
        return total_flops
    
    def count_parameters(self):
        """
        Calculate the total number of trainable parameters in the network
        
        Returns:
            Total number of parameters
        """
        total_params = 0
        in_channels = 3
        
        # Count parameters for convolutional layers
        for i, (out_channels, k_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            # Conv weights: out_channels * in_channels * k_size * k_size
            params_conv = out_channels * in_channels * k_size**2
            # Conv bias: out_channels
            params_conv += out_channels
            
            # BatchNorm parameters: 2 * out_channels (gamma and beta)
            params_bn = 2 * out_channels if self.hparams.batch_norm else 0
            
            layer_params = params_conv + params_bn
            print(f"Layer {i+1}: {layer_params} parameters")
            total_params += layer_params
            
            in_channels = out_channels
        
        # Count parameters for dense layer
        params_dense = self.feature_size * self.hparams.dense_size + self.hparams.dense_size
        print(f"Dense layer: {params_dense} parameters")
        total_params += params_dense
        
        # Count parameters for output layer
        params_output = self.hparams.dense_size * 10 + 10
        print(f"Output layer: {params_output} parameters")
        total_params += params_output
        
        return total_params
