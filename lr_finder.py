import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from model import ResNet50
from albumentation import ImageNetAlbumentations, train_transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def find_lr():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a smaller subset for LR finding
    subset_fraction = 0.05  # Use 5% of data for finding LR
    data_root = './imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
    annotations_root = './imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC'

    # Load only training data with a small subset
    train_dataset = ImageNetAlbumentations(
        root=data_root,
        annotations_root=annotations_root,
        split='train',
        transform=train_transforms,
        subset_fraction=subset_fraction
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = ResNet50(num_classes=1000).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Create optimizer with very low learning rate
    optimizer = optim.SGD(model.parameters(), 
                         lr=1e-7,  # Start with very low LR 
                         momentum=0.9,
                         weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Initialize LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    # Run range test
    print("Running LR range test...")
    lr_finder.range_test(train_loader, 
                        end_lr=10,  # End with a high LR
                        num_iter=100,  # Number of iterations to run
                        step_mode="exp",  # Exponential increase in LR
                        smooth_f=0.05)  # Smoothing factor for loss curve

    # Get the learning rates and losses
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']

    # Convert to numpy arrays for easier manipulation
    lrs = np.array(lrs)
    losses = np.array(losses)

    # Find the point of steepest descent
    gradients = np.gradient(losses)
    min_gradient_idx = np.argmin(gradients)
    
    # Find the point where the loss starts to explode
    # (defined as when the loss starts increasing rapidly)
    explosion_idx = np.argmin(losses)  # Point of minimum loss
    
    # The suggested learning rate is typically a bit lower than the point of steepest descent
    suggested_lr = lrs[min_gradient_idx] / 10  # Conservative choice

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.scatter(lrs[min_gradient_idx], losses[min_gradient_idx], color='red', 
                label='Steepest Descent')
    plt.scatter(lrs[explosion_idx], losses[explosion_idx], color='green', 
                label='Minimum Loss')
    plt.axvline(x=suggested_lr, color='orange', linestyle='--', 
                label='Suggested LR')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_finder_plot.png')
    plt.close()
    
    print(f"\nAnalysis Results:")
    print(f"Steepest descent at LR: {lrs[min_gradient_idx]:.8f}")
    print(f"Minimum loss at LR: {lrs[explosion_idx]:.8f}")
    print(f"Suggested Learning Rate: {suggested_lr:.8f}")

    # Write results to file
    with open('lr_finder_results.txt', 'w') as f:
        f.write("Learning Rate Finder Results\n")
        f.write("-" * 30 + "\n\n")
        f.write(f"Steepest descent at LR: {lrs[min_gradient_idx]:.8f}\n")
        f.write(f"Minimum loss at LR: {lrs[explosion_idx]:.8f}\n")
        f.write(f"Suggested Learning Rate: {suggested_lr:.8f}\n\n")
        f.write("Learning Rate | Loss\n")
        f.write("-" * 30 + "\n")
        for lr, loss in zip(lrs, losses):
            f.write(f"{lr:.8f} | {loss:.8f}\n")

    # Reset the model and optimizer to their initial states
    lr_finder.reset()
    
    return suggested_lr

if __name__ == "__main__":
    suggested_lr = find_lr() 