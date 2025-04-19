import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class NatureDataModule:
    def __init__(self, data_dir='./inaturalist_12K/', image_size=224, 
                 batch_size=32, num_workers=4, data_aug=True, val_split=0.2):
        """
        Data module for the iNaturalist dataset
        
        Args:
            data_dir: Directory containing the dataset
            image_size: Size to resize images to (224 for most ImageNet pre-trained models)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            data_aug: Whether to apply data augmentation
            val_split: Proportion of training data to use for validation
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug
        self.val_split = val_split
        
        # Create transforms
        self.train_transform = self._get_train_transform() if data_aug else self._get_test_transform()
        self.test_transform = self._get_test_transform()
        
        # Log configuration
        print(f"Initializing NatureDataModule:")
        print(f"- Data directory: {self.data_dir}")
        print(f"- Image size: {self.image_size}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Data augmentation: {self.data_aug}")
        
    def _get_train_transform(self):
        """Create transforms for training data with augmentation"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def _get_test_transform(self):
        """Create transforms for test/validation data"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self):
        """Set up the dataset with train/validation/test splits"""
        # Load training data
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=self.train_transform
        )
        
        # Calculate split sizes
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Load test data
        self.test_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=self.test_transform
        )
        
        # Store class names
        self.classes = train_dataset.classes
        
        print(f"Dataset setup complete:")
        print(f"- Training samples: {len(self.train_dataset)}")
        print(f"- Validation samples: {len(self.val_dataset)}")
        print(f"- Test samples: {len(self.test_dataset)}")
        print(f"- Classes: {len(self.classes)}")

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers
        )
        
    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )
        
    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )
