"""
AccelSim: Neural Accelerator Runtime Simulator

Main entry point. Runs the example MLP program with analysis and visualization.
"""

import os
from examples.mlp_program import run_example


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    run_example()
