import numpy as np
import matplotlib.pyplot as plt
import os
import json

def load_and_plot_metrics(metrics_file):
    # Load metrics from file based on extension
    if metrics_file.endswith('.json'):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    elif metrics_file.endswith('.npz'):
        metrics = dict(np.load(metrics_file))
    else:
        raise ValueError("Unsupported file format. Use .json or .npz")
    
    # Set font sizes
    plt.rcParams.update({'font.size': 12})  # Base font size
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    
    # Create figure with two subplots side by side for accuracy and loss
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use actual epoch numbers if available, otherwise use range
    epochs = metrics.get('epochs', range(1, len(metrics['train_acc']) + 1))
    
    # Plot accuracy
    ax1.plot(epochs, metrics['train_acc'], 'o-', label='Train Accuracy@1', markersize=4)
    ax1.plot(epochs, metrics['test_acc'], 'o-', label='Test Accuracy@1', markersize=4)
    ax1.plot(epochs, metrics['train_acc5'], 'o-', label='Train Accuracy@5', 
            markersize=4, alpha=0.5, color='lightblue')
    ax1.plot(epochs, metrics['test_acc5'], 'o-', label='Test Accuracy@5', 
            markersize=4, alpha=0.5, color='lightgreen')
    
    # Add target line at y=70 with text annotation
    ax1.axhline(y=70, color='black', linestyle='--', linewidth=2)
    ax1.text(30, 70.5, 'Target (70%)', 
            horizontalalignment='right', verticalalignment='bottom')
    
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(0, 75)
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
    
    # Plot loss
    ax2.plot(epochs, metrics['train_loss'], 'o-', label='Train Loss', markersize=4)
    ax2.plot(epochs, metrics['test_loss'], 'o-', label='Test Loss', markersize=4)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(0, 75)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
    
    plt.tight_layout()
    plt.savefig('metrics.png', dpi=300, bbox_inches='tight')
    
    # Create separate figure for learning rate
    fig2, ax3 = plt.subplots(figsize=(8, 8))
    
    # Plot learning rate
    ax3.plot(epochs, metrics['learning_rate'], 'o-', label='Learning Rate', markersize=4, color='purple')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.set_xlim(0, 75)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable='box')
    
    plt.tight_layout()
    plt.savefig('learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_metric_comparison(metrics_files, labels, save_dir='images'):
    """
    Compare metrics from different training runs
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy comparison
    plt.subplot(1, 3, 1)
    for metrics_file, label in zip(metrics_files, labels):
        if metrics_file.endswith('.json'):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            epochs = range(1, len(metrics['test_acc']) + 1)
            plt.plot(epochs, metrics['test_acc'], label=f'{label}')
        else:
            metrics = np.load(metrics_file)
            plt.plot(metrics['epochs'], metrics['val_acc'], label=f'{label}')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot loss comparison
    plt.subplot(1, 3, 2)
    for metrics_file, label in zip(metrics_files, labels):
        if metrics_file.endswith('.json'):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            epochs = range(1, len(metrics['test_loss']) + 1)
            plt.plot(epochs, metrics['test_loss'], label=f'{label}')
        else:
            metrics = np.load(metrics_file)
            plt.plot(metrics['epochs'], metrics['val_losses'], label=f'{label}')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate comparison
    plt.subplot(1, 3, 3)
    for metrics_file, label in zip(metrics_files, labels):
        if metrics_file.endswith('.json'):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            epochs = range(1, len(metrics['learning_rate']) + 1)
            plt.plot(epochs, metrics['learning_rate'], label=f'{label}')
        else:
            metrics = np.load(metrics_file)
            plt.plot(metrics['epochs'], metrics['lr_history'], label=f'{label}')
    plt.title('Learning Rate Schedule Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Example usage
    metrics_file = '../metrics.json'  # Changed default to .json
    
    # Plot single training run
    if os.path.exists(metrics_file):
        load_and_plot_metrics(metrics_file)
    else:
        print(f"Error: {metrics_file} not found")
    
    # Compare single run
    #metrics_files = ['../metrics.npz']
    #labels = ['Main Run']
    #if all(os.path.exists(f) for f in metrics_files):
    #    plot_metric_comparison(metrics_files, labels)