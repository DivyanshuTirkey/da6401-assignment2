import os
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class NatureDataModule(L.LightningDataModule):
    def __init__(self, data_dir='/kaggle/working/inaturalist_12K/', image_size=128, 
                 batch_size=32, num_workers=4, data_aug=True):
        """
        Data module for the iNaturalist dataset with stratified train/validation split
        
        Args:
            data_dir: Directory containing the dataset
            image_size: Size to resize images to
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            data_aug: Whether to apply data augmentation
        """
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug
        self.transforms = self._get_transforms()
        
        # Log configuration
        print(f"Initializing NatureDataModule:")
        print(f"- Data directory: {self.data_dir}")
        print(f"- Image size: {self.image_size}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Data augmentation: {self.data_aug}")
        
    def _get_transforms(self):
        """Create transforms for data preprocessing and augmentation"""
        base = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ]
        
        if self.data_aug:
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomRotation(15)
            ]
            # Log the augmentations being used
            print("Using data augmentation:")
            print(" - RandomHorizontalFlip")
            print(" - ColorJitter(0.1, 0.1, 0.1)")
            print(" - RandomRotation(15)")
            return transforms.Compose(augmentations + base)
        else:
            return transforms.Compose(base)
        
    def setup(self, stage=None):
        """Set up the dataset with stratified train/validation split"""
        print("Setting up dataset...")
        full_dataset = ImageFolder(os.path.join(self.data_dir, 'train'), 
                                  transform=self.transforms)
        
        # Log class names
        self.classes = full_dataset.classes
        print(f"Classes: {self.classes}")
        
        # Stratified split
        class_indices = {}
        for idx, (_, label) in enumerate(full_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        
        # Ensure equal class representation in the validation set
        for label, indices in class_indices.items():
            n_val = int(len(indices) * 0.2)  # 20% for validation
            np.random.shuffle(indices)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
            print(f"Class {label} ({full_dataset.classes[label]}): {len(indices)-n_val} train, {n_val} validation")
        
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = ImageFolder(os.path.join(self.data_dir, 'val'),
                                       transform=self.transforms)
        
        print(f"Total training samples: {len(self.train_dataset)}")
        print(f"Total validation samples: {len(self.val_dataset)}")
        print(f"Total test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers,
                         persistent_workers=True if self.num_workers > 0 else False)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers, 
                         persistent_workers=True if self.num_workers > 0 else False)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers)
