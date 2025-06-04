import sys
import os
from train import main as train_main, FLAGS as TrainFLAGS

if __name__ == '__main__':
    current_flags = TrainFLAGS()
    current_flags.dataset = "citeseer"
    
    # Citeseer specific flags from original run_citeseer.py
    current_flags.weight_decay = 5e-2 
    # current_flags.epochs = 200 # Example

    print(f"Running PyTorch SGLCN for Citeseer dataset with flags:")
    # Print all flags
    for flag_name, value in current_flags.__dict__.items():
         if not flag_name.startswith('__'):
            print(f"--{flag_name} {value}")
            
    train_main(current_flags)