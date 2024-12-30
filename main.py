import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentation import ImageNetAlbumentations, train_transforms, val_transforms
from model import ResNet50
from utils import train, test, print_mps_memory
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.mps
from torch.amp import autocast, GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

main_dir = "/mnt/imagenet/assignment/"

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """Save checkpoint and best model"""
    if not os.path.exists(main_dir + checkpoint_dir):
        os.makedirs(main_dir + checkpoint_dir)
    
    filepath = os.path.join(main_dir, checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(main_dir, checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def main():
    try:
        # Create checkpoint directory
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(main_dir + checkpoint_dir):
            os.makedirs(main_dir + checkpoint_dir)
        
        # Set device and configure device-specific settings
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Optimize for fixed input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Print GPU info
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Hyperparameters optimized for 16GB GPU
            batch_size = 512  # Increased for faster training
            num_workers = 8   # Adjust based on CPU cores
            pin_memory = True
            prefetch_factor = 2  # Prefetch 2 batches per worker
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            if hasattr(torch.backends.mps, 'enable_async_copy'):
                torch.backends.mps.enable_async_copy()
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        # Hyperparameters - adjusted for different devices
        if device.type == "cuda":
            batch_size = 256  # Can be larger on GPU
            num_workers = 8   # More workers for GPU
            pin_memory = True
        elif device.type == "mps":
            batch_size = 128  # Reduced for MPS
            num_workers = 12   # Fewer workers for MacBook
            pin_memory = True #False
        else:
            batch_size = 64   # Smallest for CPU
            num_workers = 2
            pin_memory = False
        
        epochs = 90
        base_lr = 0.05
        momentum = 0.9
        weight_decay = 1e-4

        # Model
        model = ResNet50(num_classes=1000).to(device, memory_format=torch.channels_last)  # Use channels_last memory format
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = DDP(model)  # Use DistributedDataParallel instead of DataParallel
        
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
        subset_fraction = 1.0
        data_root = main_dir + 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
        annotations_root = main_dir + 'imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC'
        
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
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            generator=torch.Generator(device='cpu')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            generator=torch.Generator(device='cpu')
        )
        
        # Print model summary and parameters using torchinfo
        print(summary(model, input_size=(1, 3, 224, 224), device=device))
        
        # Training state
        start_epoch = 1
        best_acc = 0
        
        # Resume from checkpoint if exists
        checkpoint_path = os.path.join(main_dir, checkpoint_dir, 'checkpoint.pth.tar')
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
        
        # Memory management for different devices
        if device.type == 'cuda':
            # Enable automatic mixed precision for CUDA
            scaler = GradScaler('cuda')
        elif device.type == 'mps':
            def clear_mps_cache():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            import atexit
            atexit.register(clear_mps_cache)
        
        # Training loop
        for epoch in range(start_epoch, epochs + 1):
            # Clear cache if needed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            print(f'\nEpoch: {epoch}')
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {current_lr:.6f}')
            
            # Training phase
            train_loss, train_accuracy = train(model, device, train_loader, 
                                             optimizer, criterion, epoch,
                                             scaler=scaler if device.type == 'cuda' else None)
            
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
            
            # Print memory stats
            if device.type == 'cuda':
                print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
                      f"(Max: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")
            elif device.type == 'mps':
                print_mps_memory()
        
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
        plt.savefig(main_dir + 'images/training_curves.png')
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
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