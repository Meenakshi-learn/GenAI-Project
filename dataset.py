import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import random

class FisheyeCorrectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        """
        Args:
            root_dir: Path to directory containing paired images
            transform: Transforms to apply to images
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        
        # Find all image files recursively
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(self.root_dir.rglob(ext))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid image if loading fails
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Get image dimensions
        width, height = image.size
        
        # Split the combined image (distorted | clean)
        # Assumes images are stacked horizontally
        if width >= height * 2:
            # Horizontal stacking: [distorted | clean]
            distorted = image.crop((0, 0, width // 2, height))
            clean = image.crop((width // 2, 0, width, height))
        else:
            # Vertical stacking: [distorted / clean]
            distorted = image.crop((0, 0, width, height // 2))
            clean = image.crop((0, height // 2, width, height))
        
        # Convert to tensors if no custom transform
        if self.transform is None:
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            distorted = to_tensor(distorted)
            clean = to_tensor(clean)
            
            distorted = normalize(distorted)
            clean = normalize(clean)
        else:
            distorted = self.transform(distorted)
            clean = self.transform(clean)
        
        return distorted, clean, str(img_path)

def get_dataloaders(processed_dir, batch_size=16, num_workers=4, val_split=0.1):
    """Create train and validation dataloaders"""
    
    # Define transforms
    train_transform = None  # We'll use default transforms
    val_transform = None
    
    # Create full dataset
    full_dataset = FisheyeCorrectionDataset(processed_dir, transform=train_transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset
    dataset = FisheyeCorrectionDataset("data/processed")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        distorted, clean, path = dataset[0]
        print(f"Sample shape - Distorted: {distorted.shape}, Clean: {clean.shape}")
        print(f"Sample path: {path}")