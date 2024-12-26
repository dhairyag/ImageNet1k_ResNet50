import os
from pathlib import Path

class ImageNetStructure:
    """Helper class to organize ImageNet data structure"""
    
    def __init__(self, root_dir='./data/imagenet'):
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / 'train'
        self.val_dir = self.root_dir / 'val'
        
    def create_directories(self):
        """Create the necessary directory structure"""
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        
    def check_structure(self):
        """Check if the ImageNet directory structure is correct"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"ImageNet root directory not found at {self.root_dir}")
            
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found at {self.train_dir}")
            
        if not self.val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found at {self.val_dir}")
            
        # Check if there are 1000 class directories in training
        train_subdirs = list(self.train_dir.glob('*'))
        if len(train_subdirs) != 1000:
            raise ValueError(f"Expected 1000 class directories in train, found {len(train_subdirs)}")
            
        # Check if there are 1000 class directories in validation
        val_subdirs = list(self.val_dir.glob('*'))
        if len(val_subdirs) != 1000:
            raise ValueError(f"Expected 1000 class directories in validation, found {len(val_subdirs)}")
            
        print("✓ ImageNet directory structure is correct")
        
    def print_structure(self):
        """Print the directory structure and some statistics"""
        print(f"\nImageNet Directory Structure:")
        print(f"Root: {self.root_dir}")
        print(f"Training: {self.train_dir}")
        print(f"Validation: {self.val_dir}")
        
        train_classes = len(list(self.train_dir.glob('*')))
        val_classes = len(list(self.val_dir.glob('*')))
        
        print(f"\nStatistics:")
        print(f"Training classes: {train_classes}")
        print(f"Validation classes: {val_classes}")
        
        # Count total images
        train_images = sum(len(list(d.glob('*.JPEG'))) for d in self.train_dir.glob('*'))
        val_images = sum(len(list(d.glob('*.JPEG'))) for d in self.val_dir.glob('*'))
        
        print(f"Training images: {train_images:,}")
        print(f"Validation images: {val_images:,}")

def main():
    """Main function to set up ImageNet directory structure"""
    imagenet = ImageNetStructure()
    imagenet.create_directories()
    
    print("\nPlease organize your ImageNet data as follows:")
    print("./data/imagenet/")
    print("├── train/")
    print("│   ├── n01440764/")
    print("│   ├── n01443537/")
    print("│   └── ... (1000 class folders)")
    print("└── val/")
    print("    ├── n01440764/")
    print("    ├── n01443537/")
    print("    └── ... (1000 class folders)")
    
    try:
        imagenet.check_structure()
        imagenet.print_structure()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("1. Downloaded ImageNet ILSVRC2012 dataset")
        print("2. Extracted the archives")
        print("3. Organized the data in the correct directory structure")

if __name__ == '__main__':
    main() 