import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from main import main

def run_distributed(world_size, args):
    # Set environment variables for distributed training
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK'] = '0'

    mp.spawn(
        main,
        args=(args,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=None, 
                      help='Number of GPUs to use (default: all available)')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size per GPU')
    args = parser.parse_args()

    # Setup distributed environment
    world_size = args.num_gpus or torch.cuda.device_count()
    print(f"Starting training on {world_size} GPU(s)...")
    
    if world_size > 1:
        run_distributed(world_size, args)
    else:
        # For single GPU, run without distributed setup
        main(args) 