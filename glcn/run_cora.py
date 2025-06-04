import sys
import os
# Assuming train_pytorch.py and FLAGS are structured to be importable or run directly
# For simplicity, this script will set the dataset flag and call the main training function.

# If train_pytorch.py is directly executable and parses args (e.g. with argparse)
# you would use:
# os.system(f"python train_pytorch.py --dataset cora")

# If train_pytorch.py has a main(flags_obj) function:
from train import main as train_main, FLAGS as TrainFLAGS

if __name__ == '__main__':
    current_flags = TrainFLAGS()
    current_flags.dataset = "cora"
    
    # You can override other Cora-specific flags here if needed
    # current_flags.epochs = 200 # Example
    
    print(f"Running PyTorch SGLCN for Cora dataset with default flags:")
    # Print all flags
    for flag_name, value in current_flags.__dict__.items():
         if not flag_name.startswith('__'):
            print(f"--{flag_name} {value}")

    train_main(current_flags)