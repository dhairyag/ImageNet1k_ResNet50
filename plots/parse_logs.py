import re
import json

def parse_log_file(log_file_path):
    """
    Parse training log file and extract metrics.
    Returns dictionary with lists of metrics.
    """
    metrics = {
        'epochs': [],       # Epoch numbers
        'train_acc': [],    # Acc@1 from training
        'train_acc5': [],   # Acc@5 from training
        'train_loss': [],   # Loss from training
        'test_acc': [],     # Acc@1 from test
        'test_acc5': [],    # Acc@5 from test
        'test_loss': [],    # Loss from test
        'learning_rate': [] # Learning rate
    }
    
    # Regex patterns
    epoch_pattern = r"Epoch: (\d+)"
    train_pattern = r"Train Epoch: \d+ Loss: ([\d.]+) Acc@1 ([\d.]+) Acc@5 ([\d.]+)"
    test_pattern = r"Test set: Average loss: ([\d.]+), Acc@1: ([\d.]+)%, Acc@5: ([\d.]+)%"
    lr_pattern = r"Learning rate: ([\d.]+)"
    
    current_epoch = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Extract epoch number
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Extract training metrics
            train_match = re.search(train_pattern, line)
            if train_match and current_epoch is not None:
                metrics['epochs'].append(current_epoch)
                metrics['train_loss'].append(float(train_match.group(1)))
                metrics['train_acc'].append(float(train_match.group(2)))
                metrics['train_acc5'].append(float(train_match.group(3)))
            
            # Extract test metrics
            test_match = re.search(test_pattern, line)
            if test_match:
                metrics['test_loss'].append(float(test_match.group(1)))
                metrics['test_acc'].append(float(test_match.group(2)))
                metrics['test_acc5'].append(float(test_match.group(3)))
            
            # Extract learning rate
            lr_match = re.search(lr_pattern, line)
            if lr_match:
                metrics['learning_rate'].append(float(lr_match.group(1)))
    
    return metrics

def save_metrics(metrics, output_file):
    """Save metrics to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    # Example usage
    log_file = '../log_output.md'
    output_file = '../metrics.json'
    
    metrics = parse_log_file(log_file)
    save_metrics(metrics, output_file)
    
    # Print some statistics
    print(f"Extracted metrics for {len(metrics['train_acc'])} epochs")
    print(f"Final training accuracy: {metrics['train_acc'][-1]:.2f}%")
    print(f"Final test accuracy: {metrics['test_acc'][-1]:.2f}%") 