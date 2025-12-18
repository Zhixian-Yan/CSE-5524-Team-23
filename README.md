# DINOv2 Inference Script

This repository provides an inference script for plant species classification using DINOv2 models. The project implements a **frozen backbone + lightweight classification head** approach, where we train lightweight classification heads (Linear and MLP) on top of a frozen DINOv2 backbone for efficient and stable transfer learning.

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch timm pandas Pillow
```

### Using conda

#### Option 1: Create a new conda environment

```bash
# Create a new conda environment with Python 3.8+
conda create -n dinov2 python=3.8

# Activate the environment
conda activate dinov2

# Install PyTorch (choose the appropriate version for your system)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
conda install -c conda-forge timm pandas pillow
```

#### Option 2: Install from requirements.txt in conda environment

```bash
# Create and activate conda environment
conda create -n dinov2 python=3.8
conda activate dinov2

# Install PyTorch first (recommended via conda)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining packages via pip
pip install timm pandas Pillow
```

#### Option 3: Use conda environment.yml (if available)

If you have an `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate dinov2
```

## Project Structure

```
.
├── infer_dinov2_head.py          # Main inference script
├── requirements.txt               # Python dependencies
├── DINOv2/
│   ├── checkpoint/
│   │   └── model_best.pth.tar    # DINOv2 backbone checkpoint
│   └── dataset/
│       └── species_ids.csv       # Species ID mapping file
├── outputs/
│   ├── dino_linear_head.pth      # Linear classification head model
│   └── dino_mlp_head.pth         # MLP classification head model
└── examples/                      # Example images
    ├── example_1.jpg
    ├── example_2.jpg
    └── example_3.jpg
```

## Usage

### Basic Usage

Inference with Linear Head:

```bash
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_linear_head.pth \
    --head-type linear
```

Inference with MLP Head:

```bash
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_mlp_head.pth \
    --head-type mlp
```

### Running with conda

If you're using conda, make sure to activate your environment first:

```bash
# Activate conda environment
conda activate dinov2

# Run inference
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_linear_head.pth \
    --head-type linear
```

### Arguments

- `--img`: **Required**, path to input image
- `--species-ids`: **Required**, path to species ID CSV file (must contain `species_id` column)
- `--checkpoint`: **Required**, path to DINOv2 backbone checkpoint
- `--head-pth`: **Required**, path to classification head model file
- `--head-type`: **Required**, type of classification head, options: `linear` or `mlp`
- `--device`: **Optional**, computing device, default is `cuda`. Set to `cpu` if using CPU

### CPU Inference

If you don't have a GPU or want to use CPU:

```bash
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_linear_head.pth \
    --head-type linear \
    --device cpu
```

## Output

The script will output:
- Predicted class index
- Predicted species_id

Example output:
```
Predicted class index: 42
Predicted species_id: 1356115
```

## Model Architecture

### Overview

This project uses a **frozen DINOv2 backbone** with trainable **classification heads** for plant species classification. This design leverages powerful pretrained representations while keeping training fast and efficient.

### Backbone (Frozen)

- **Model**: DINOv2 ViT-B/14 (`vit_base_patch14_reg4_dinov2.lvd142m`)
- **Status**: Pretrained and **fully frozen** during training
- **Output**: 768-dimensional feature vector per image
- **Gradients**: No backpropagation into the backbone

**Benefits:**
- Stable representations from pretrained model
- Fast training (only head parameters updated)
- Low risk of overfitting on limited labeled data
- Minimal computational requirements

### Classification Heads

We provide two types of classification heads:

#### Linear Head

A single fully connected layer that serves as a strong baseline:

```
Input (768-d) → Linear Layer → Output (98 classes)
```

- **Architecture**: `Linear(768 → 98)`
- **Parameters**: ~75K trainable parameters
- **Use case**: Strong baseline for evaluating representation quality
- **Training**: Cross-Entropy Loss, Adam optimizer, 5 epochs

#### MLP Head

A multi-layer perceptron with one hidden layer for non-linear decision boundaries:

```
Input (768-d) → Linear(768 → 1024) → ReLU → Linear(1024 → 98) → Output
```

- **Architecture**: 
  - `Linear(768 → 1024)`
  - `ReLU` activation
  - `Linear(1024 → 98)`
- **Parameters**: ~1M trainable parameters
- **Use case**: Can model more complex class boundaries, useful for larger datasets
- **Training**: Same setup as linear head (Cross-Entropy Loss, Adam optimizer, 5 epochs)

### Training Approach

**Why Frozen Backbone + Lightweight Head?**

This approach is widely adopted in modern vision research because it:

- ✅ Leverages powerful pretrained representations without fine-tuning
- ✅ Requires minimal computational resources
- ✅ Is easy to reproduce and deploy
- ✅ Separates representation learning from task-specific classification
- ✅ Reduces overfitting risk on small datasets

Both heads were trained for 5 epochs and evaluated on a held-out validation set. The trained models are provided for inference.

## File Descriptions

### Required Files

1. **DINOv2 Checkpoint** (`DINOv2/checkpoint/model_best.pth.tar`)
   - Pre-trained weights for DINOv2 backbone
   - Model architecture: `vit_base_patch14_reg4_dinov2.lvd142m`

2. **Classification Head Model** (`outputs/dino_linear_head.pth` or `outputs/dino_mlp_head.pth`)
   - Trained classification head weights
   - Select the corresponding file based on `--head-type` argument

3. **Species ID File** (`DINOv2/dataset/species_ids.csv`)
   - CSV format, must contain `species_id` column
   - Used to map predicted class indices to actual species IDs

## Notes

1. Ensure all required files exist and paths are correct
2. If using GPU, make sure you have installed CUDA-enabled PyTorch
3. Images are automatically converted to RGB format and preprocessed as needed
4. Models are automatically set to evaluation mode during inference

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```
   Or with conda:
   ```bash
   conda activate dinov2
   pip install -r requirements.txt
   ```

2. **CUDA out of memory**: If GPU memory is insufficient, use CPU mode
   ```bash
   --device cpu
   ```

3. **File not found**: Check that file paths are correct, use absolute or relative paths

4. **Model loading error**: Ensure checkpoint files and head files are in the correct format and match the model architecture in the code

5. **Conda environment issues**: If you encounter issues with conda:
   ```bash
   # Verify conda environment is activated
   conda info --envs
   
   # Check Python version
   python --version
   
   # Verify packages are installed
   conda list
   ```
