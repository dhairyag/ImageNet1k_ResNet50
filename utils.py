import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from train import AverageMeter, accuracy

###################
# Utility Functions
###################

def visualize_augmentations(dataset, samples=36):
    import matplotlib.pyplot as plt
    import random
    
    plt.figure(figsize=(10, 10))
    for i in range(samples):
        idx = random.randint(0, len(dataset)-1)
        data = dataset[idx][0]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if data.shape[0] == 1:  # If channels first, move to last
            data = np.transpose(data, (1, 2, 0))
        plt.subplot(6, 6, i + 1)
        plt.imshow(data.squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader)
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    losses = AverageMeter('Loss')
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        
        loss.backward()
        optimizer.step()
        
        pbar.set_description(
            f'Train Epoch: {epoch} Loss: {losses.avg:.3f} '
            f'Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        )
    
    return losses.avg, top1.avg

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    losses = AverageMeter('Loss')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # Measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
    
    print(f'\nTest set: Average loss: {losses.avg:.4f}, '
          f'Acc@1: {top1.avg:.3f}%, Acc@5: {top5.avg:.3f}%\n')
    
    return losses.avg, top1.avg

