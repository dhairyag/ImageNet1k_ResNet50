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
from torch.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np

main_dir = os.getcwd()

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """Save checkpoint and best model"""
    if not os.path.exists(main_dir + checkpoint_dir):
        os.makedirs(main_dir + checkpoint_dir)
    
    filepath = os.path.join(main_dir, checkpoint_dir, filename)
    # Save without weights_only parameter
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(main_dir, checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def setup_distributed(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def main(rank=0, args=None):
    try:
        # Set device first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize process group only for multi-GPU setup
        is_distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1
        if is_distributed:
            setup_distributed(rank, torch.cuda.device_count())
            device = torch.device(f"cuda:{rank}")
            is_main_process = rank == 0
        else:
            is_main_process = True

        # Set hyperparameters first
        # Set default parameters based on device
        if device.type == "cuda":
            num_workers = 8
            pin_memory = True
            prefetch_factor = 2
            default_batch_size = 256
        elif device.type == "mps":
            num_workers = 12
            pin_memory = True
            prefetch_factor = 2
            default_batch_size = 128
        else:
            num_workers = 2
            pin_memory = False
            prefetch_factor = 2
            default_batch_size = 64

        # Override batch_size if provided in args
        batch_size = args.batch_size if args and hasattr(args, 'batch_size') else default_batch_size

        # Model setup
        model = ResNet50(num_classes=1000).to(device, memory_format=torch.channels_last)
        
        # Wrap model in DDP only for multi-GPU
        if is_distributed:
            model = DDP(model, device_ids=[rank])

        # Data loading
        subset_fraction = 1.0
        data_root = main_dir + '/../ILSVRC/Data/CLS-LOC'
        annotations_root = main_dir + '/../ILSVRC/Annotations/CLS-LOC'
        
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
        
        # Data loading with DistributedSampler only for distributed training
        train_sampler = None
        val_sampler = None
        if is_distributed:  # Only use samplers for multi-GPU
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),  # Shuffle if no sampler
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
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
        
        # Print model summary and parameters using torchinfo
        print(summary(model, input_size=(1, 3, 224, 224), device=device))
        
        # Training state
        start_epoch = 1
        best_acc = 0
        
        # Resume from checkpoint if exists
        checkpoint_path = os.path.join(main_dir, checkpoint_dir, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            # Load checkpoint without weights_only parameter
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )
            
            # Load model weights
            model.load_state_dict(checkpoint['state_dict'])
            
            # Safely load other state information
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc = checkpoint.get('best_acc', 0)
            
            # Load optimizer and scheduler states if available
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                
            # Load history if available
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_acc = checkpoint.get('train_acc', [])
            val_acc = checkpoint.get('val_acc', [])
            lr_history = checkpoint.get('lr_history', [])
            
            print(f"=> loaded checkpoint 'epoch {checkpoint.get('epoch', 0)}' "
                  f"(acc {checkpoint.get('best_acc', 0):.2f})")
        
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
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
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
            if is_main_process:
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
                if is_main_process:
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
            
            # Save metrics to numpy file
            if is_main_process:
                # Convert tensors to CPU before saving
                metrics = {
                    'train_losses': [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses],
                    'val_losses': [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in val_losses],
                    'train_acc': [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_acc],
                    'val_acc': [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in val_acc],
                    'lr_history': [lr.cpu().item() if torch.is_tensor(lr) else lr for lr in lr_history],
                    'epochs': list(range(1, epoch + 1))
                }
                metrics_path = os.path.join(main_dir, 'metrics.npz')
                np.savez(metrics_path, **metrics)
        
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
        plt.savefig(main_dir + '/images/training_curves.png')
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                dist.destroy_process_group()
            except:
                pass
        raise  # Re-raise the exception after cleanup

    finally:
        # Clean up
        if 'train_loader' in locals():
            train_loader._iterator = None
        if 'val_loader' in locals():
            val_loader._iterator = None
        plt.close('all')
        if is_distributed:
            try:
                dist.destroy_process_group()
            except:
                pass

if __name__ == '__main__':
    main()  # Call without args when running directly 