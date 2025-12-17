#!/usr/bin/env python3
"""
Standalone LoRA fine-tuning script for BioCLIP 2
Can be run without SLURM for local/single GPU training
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.main import main

if __name__ == "__main__":
    # Example command line arguments for LoRA fine-tuning
    # Modify these according to your needs
    args = [
        "--train-data", "[your-train-data-path]",  # Update this
        "--val-data", "[your-val-data-path]",      # Update this
        "--dataset-type", "webdataset",
        "--pretrained", "imageomics/bioclip-2",    # BioCLIP 2 model
        "--model", "ViT-L-14",
        "--batch-size", "32",
        "--accum-freq", "1",
        "--epochs", "10",
        "--workers", "4",
        "--lr", "1e-4",
        "--warmup", "100",
        "--seed", "42",
        "--precision", "bf16",
        "--logs", "./logs",
        "--save-frequency", "1",
        "--log-every-n-steps", "10",
        "--use-lora",
        "--lora-rank", "8",
        "--lora-alpha", "16.0",
        "--lora-dropout", "0.1",
        "--lora-enable-vision",
        "--lora-enable-text",
    ]
    
    main(args)


