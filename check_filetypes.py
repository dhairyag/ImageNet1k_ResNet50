from pathlib import Path
from collections import Counter
import os

def check_file_types(directory):
    """Check all file types in the directory and its subdirectories"""
    extensions = Counter()
    
    # Walk through all directories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the file extension
            ext = os.path.splitext(file)[1].lower()
            if ext:  # Only count if there is an extension
                extensions[ext] += 1
                
    return extensions

def main():
    data_dir = Path('./data/tiny-imagenet-200')
    
    print("Checking train directory...")
    train_extensions = check_file_types(data_dir / 'train')
    print("\nFile extensions in train:")
    for ext, count in train_extensions.most_common():
        print(f"{ext}: {count}")
        
    print("\nChecking validation directory...")
    val_extensions = check_file_types(data_dir / 'val')
    print("\nFile extensions in val:")
    for ext, count in val_extensions.most_common():
        print(f"{ext}: {count}")

if __name__ == '__main__':
    main() 