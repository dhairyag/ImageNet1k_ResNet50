import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import random
import xml.etree.ElementTree as ET

# ImageNet mean and std values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms for ImageNet
train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# Validation transforms for ImageNet
val_transforms = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

class ImageNetAlbumentations:
    def __init__(self, root, annotations_root, split='train', transform=None, subset_fraction=0.01):
        """
        Args:
            root: Root directory of the ImageNet dataset images
            annotations_root: Root directory of the annotations
            split: 'train' or 'val'
            transform: Albumentations transforms to apply
            subset_fraction: Fraction of data to use (0.01 = 1%)
        """
        self.root = Path(root)
        self.annotations_root = Path(annotations_root)
        self.split = split
        self.transform = transform
        self.subset_fraction = subset_fraction
        self.samples = []
        
        # Setup directories
        split_dir = self.root / split
        annotations_dir = self.annotations_root / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {split_dir}")
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
            
        print(f"\nScanning {split} directory: {split_dir}")
        print(f"Using annotations from: {annotations_dir}")
        
        if split == 'train':
            self._load_train_data(split_dir)
        else:
            self._load_val_data(split_dir, annotations_dir)
            
        if not self.samples:
            raise RuntimeError(f"Found 0 files in {split_dir}")
        
        # Randomly select subset of data
        num_samples = len(self.samples)
        subset_size = int(num_samples * subset_fraction)
        self.samples = random.sample(self.samples, subset_size)
        
        print(f"\nTotal loaded {len(self.samples)} images "
              f"({subset_fraction*100:.1f}%) from {split_dir}")
    
    def _get_class_from_xml(self, xml_path):
        """Extract class name from XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return root.find('.//name').text
    
    def _load_train_data(self, split_dir):
        """Load training data with class subdirectories"""
        class_dirs = sorted(split_dir.glob('*'))
        print(f"Found {len(class_dirs)} class directories")
        
        # Create class to index mapping
        self.class_to_idx = {class_dir.name: idx 
                            for idx, class_dir in enumerate(class_dirs)}
        
        for class_dir in class_dirs:
            if class_dir.is_dir():
                class_samples = []
                for ext in ['*.JPEG', '*.jpeg', '*.jpg', '*.JPG']:
                    class_samples.extend(class_dir.glob(ext))
                
                if class_samples:
                    class_idx = self.class_to_idx[class_dir.name]
                    self.samples.extend([(str(img_path), class_idx) 
                                      for img_path in class_samples])
                    print(f"Class {class_dir.name}: Found {len(class_samples)} images")
                else:
                    print(f"Warning: No images found in {class_dir}")
    
    def _load_val_data(self, split_dir, annotations_dir):
        """Load validation data using XML annotations"""
        print("Reading validation annotations...")
        
        # Create class to index mapping if not already created
        if not hasattr(self, 'class_to_idx'):
            train_class_dirs = sorted((self.root / 'train').glob('*'))
            self.class_to_idx = {class_dir.name: idx 
                               for idx, class_dir in enumerate(train_class_dirs)}
        
        # Load all validation images and their annotations
        all_images = []
        for ext in ['*.JPEG', '*.jpeg', '*.jpg', '*.JPG']:
            all_images.extend(split_dir.glob(ext))
        
        # Create samples list with proper class indices
        for img_path in all_images:
            # Get corresponding XML file
            xml_path = annotations_dir / f"{img_path.stem}.xml"
            if xml_path.exists():
                try:
                    class_name = self._get_class_from_xml(xml_path)
                    if class_name in self.class_to_idx:
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    else:
                        print(f"Warning: Unknown class {class_name} for image {img_path.name}")
                except Exception as e:
                    print(f"Warning: Failed to parse annotation for {img_path.name}: {e}")
            else:
                print(f"Warning: No annotation found for image {img_path.name}")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
            
        return img, label
