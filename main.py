import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentation import ImageNetAlbumentations, train_transforms, val_transforms
from model import ResNet50
from utils import train, test
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """Save checkpoint and best model"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def main():
    try:
        # Create checkpoint directory
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Hyperparameters
        batch_size = 256
        epochs = 90
        base_lr = 0.1  # Lower initial LR due to smaller dataset
        momentum = 0.9
        weight_decay = 1e-4
        
        # Model
        model = ResNet50(num_classes=1000).to(device)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=momentum, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup and cosine decay
        warmup_epochs = 5
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        # Data loading
        subset_fraction = 0.04  # Use 1% of the data
        data_root = './imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
        annotations_root = './imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC'
        
        train_dataset = ImageNetAlbumentations(
            root=data_root,
            annotations_root=annotations_root,
            split='train',
            transform=train_transforms,
            subset_fraction=subset_fraction
        )
        
        val_dataset = ImageNetAlbumentations(
            root=data_root,
            annotations_root=annotations_root,
            split='val',
            transform=val_transforms,
            subset_fraction=subset_fraction
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Reduced from 8 to 4 for better stability
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Reduced from 8 to 4
            pin_memory=True,
            persistent_workers=True
        )
        
        # Print model summary and parameters using torchinfo
        print(summary(model, input_size=(1, 3, 224, 224), device=device))
        
        # Training state
        start_epoch = 1
        best_acc = 0
        
        # Resume from checkpoint if exists
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint 'epoch {checkpoint['epoch']}' (acc {checkpoint['best_acc']:.2f})")
        
        # Training and testing
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []
        lr_history = []
        
        # Run LR finder if needed
        # if args.find_lr:  # Add argument parser for this
        #     from lr_finder import find_lr
        #     suggested_lr = find_lr()
        #     base_lr = suggested_lr / 10  # Conservative starting point
        # else:
        #     base_lr = 0.01  # Default value
        
        for epoch in range(start_epoch, epochs + 1):
            print(f'\nEpoch: {epoch}')
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {current_lr:.6f}')
            
            # Training phase
            train_loss, train_accuracy = train(model, device, train_loader, 
                                             optimizer, criterion, epoch)
            
            # Validation phase
            val_loss, val_accuracy = test(model, device, val_loader, criterion)
            
            # Step the scheduler
            scheduler.step()
            
            # Save metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            lr_history.append(current_lr)
            
            # Remember best accuracy and save checkpoint
            is_best = val_accuracy > best_acc
            best_acc = max(val_accuracy, best_acc)
            
            # Save checkpoint every epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr_history': lr_history
            }, is_best)
            
            # Save additional checkpoint every 10 epochs
            if epoch % 10 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'lr_history': lr_history
                }, False, filename=f'checkpoint_epoch_{epoch}.pth.tar')
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        plt.plot(lr_history)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        #plt.show()
        
        # Save the plot as a PNG file
        plt.savefig('images/training_curves.png')
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        # Save emergency checkpoint on error
        if 'model' in locals():
            save_checkpoint({
                'epoch': epoch if 'epoch' in locals() else 0,
                'state_dict': model.state_dict(),
                'best_acc': best_acc if 'best_acc' in locals() else 0,
                'optimizer': optimizer.state_dict() if 'optimizer' in locals() else None,
                'scheduler': scheduler.state_dict() if 'scheduler' in locals() else None,
                'train_losses': train_losses if 'train_losses' in locals() else [],
                'val_losses': val_losses if 'val_losses' in locals() else [],
                'train_acc': train_acc if 'train_acc' in locals() else [],
                'val_acc': val_acc if 'val_acc' in locals() else [],
                'lr_history': lr_history if 'lr_history' in locals() else []
            }, False, filename='emergency_checkpoint.pth.tar')
    finally:
        # Clean up
        if 'train_loader' in locals():
            train_loader._iterator = None
        if 'val_loader' in locals():
            val_loader._iterator = None
        plt.close('all')

if __name__ == '__main__':
    main() 