import torch
import torch.nn as nn
from torchvision import models

class PretrainedModel(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=10, dropout=0.2, freeze_strategy='feature_extraction'):
        """
        Fine-tuning pre-trained models for iNaturalist dataset
        
        Args:
            model_name: Name of the pre-trained model to use
            num_classes: Number of classes in the dataset
            dropout: Dropout rate for the classifier
            freeze_strategy: Strategy for freezing layers ['feature_extraction', 'partial_finetuning', 'progressive_unfreezing']
        """
        super(PretrainedModel, self).__init__()
        
        # Load pre-trained model
        self.model = self._load_pretrained_model(model_name, num_classes, dropout)
        self.freeze_strategy = freeze_strategy
        self.current_epoch = 0
        
        # Apply freezing strategy
        self.apply_freeze_strategy(freeze_strategy)
        
    def _load_pretrained_model(self, model_name, num_classes, dropout):
        """Load a pre-trained model and modify the classifier for our dataset"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Replace the final fully connected layer
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes)
            )
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            # Replace classifier
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            # Replace classifier
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout, inplace=True),
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
                
        return model
    
    def apply_freeze_strategy(self, strategy, current_epoch=None):
        """Apply the specified freezing strategy to the model"""
        if current_epoch is not None:
            self.current_epoch = current_epoch
            
        if strategy == 'feature_extraction':
            # Freeze all layers except the final classifier
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Unfreeze the final layer
            if hasattr(self.model, 'fc'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, 'classifier'):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
                    
        elif strategy == 'partial_finetuning':
            # For ResNet, freeze first 7 layer groups
            if isinstance(self.model, models.ResNet):
                children = list(self.model.children())
                # Freeze first 7 layer groups
                for i in range(7):
                    for param in children[i].parameters():
                        param.requires_grad = False
                        
            # For VGG, freeze features
            elif isinstance(self.model, models.VGG):
                for param in self.model.features.parameters():
                    param.requires_grad = False
                    
            # For EfficientNet, freeze features
            elif 'EfficientNet' in self.model.__class__.__name__:
                for param in self.model.features.parameters():
                    param.requires_grad = False
                    
        elif strategy == 'progressive_unfreezing':
            # Progressive unfreezing based on current epoch
            if self.current_epoch == 0:
                # Start with all layers frozen except the last
                self.apply_freeze_strategy('feature_extraction')
                
            elif self.current_epoch == 5:
                # Unfreeze the last convolutional block
                if isinstance(self.model, models.ResNet):
                    for param in self.model.layer4.parameters():
                        param.requires_grad = True
                elif isinstance(self.model, models.VGG):
                    features = list(self.model.features.children())
                    for layer in features[-4:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                elif 'EfficientNet' in self.model.__class__.__name__:
                    features = list(self.model.features.children())
                    for layer in features[-3:]:
                        for param in layer.parameters():
                            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
    
    def count_trainable_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self):
        """Count the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
