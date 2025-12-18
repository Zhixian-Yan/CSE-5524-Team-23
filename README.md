# Final Project – PlantCLEF 2025 Team-23

This repository contains multiple approaches for plant species classification and prediction, including task-aware prediction filtering, DINOv2-based classification, BioCLIP-2 fine-tuning, and other experimental models.

## Table of Contents

1. [Installation](#installation)
2. [Projects Overview](#projects-overview)
   - [Task-aware Prediction Filtering](#1-task-aware-prediction-filtering)
   - [DINOv2 Linear/MLP Classification](#2-dinov2-linearmlp-classification)
   - [BioCLIP-2 LoRA Fine-tuning](#3-bioclip-2-lora-fine-tuning)
   - [Other Experimental Models](#4-other-experimental-models)

---

## Installation

### General Requirements

- **Python 3.11** (recommended for main project)
- **Python 3.8+** (for DINOv2 project)

### Quick Setup

```bash
# Install main dependencies
pip install -r requirements.txt
```

For specific projects, refer to their respective sections below.

---

## Projects Overview

This repository contains four main approaches to plant species classification:

### 1. Task-aware Prediction Filtering

**Location**: `Task-aware Prediction Filtering/`

This project implements task-aware prediction filtering for PlantCLEF 2025 quadrat prediction.

#### Pretrained Model

Download the pretrained model and metadata:
- **Pretrained Model**: [Download from Google Drive](https://drive.google.com/file/d/15Yxi9vovUxo4YUMWYcO2k3JqzFPQgEBm/view?usp=sharing)
- **PlantCLEF2024 Metadata**: [Download from Google Drive](https://drive.google.com/file/d/1z3gx4W6Vj9iK0V-o5pS96rcIZeBShuPX/view?usp=sharing)

Save them to the `Task-aware Prediction Filtering/` folder.

#### Test Examples

Validation images are provided under `Task-aware Prediction Filtering/test_imgs/`.

#### Usage

Run the Jupyter notebook to see PlantCLEF 2025 quadrat prediction:

```bash
cd "Task-aware Prediction Filtering"
jupyter notebook Tile_inference.ipynb
```

---

### 2. DINOv2 Linear/MLP Classification

**Location**: `DINOv2-linear-mlp/`

This project implements plant species classification using DINOv2 models with lightweight classification heads (Linear and MLP). The approach uses a **frozen backbone + lightweight classification head** for efficient and stable transfer learning.

#### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

#### Installation

**Using pip:**
```bash
cd DINOv2-linear-mlp
pip install -r requirements.txt
```

**Using conda:**
```bash
# Create conda environment
conda create -n dinov2 python=3.8
conda activate dinov2

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install timm pandas Pillow
```

#### Model Architecture

- **Backbone**: DINOv2 ViT-B/14 (frozen, pretrained)
- **Classification Heads**: 
  - **Linear Head**: Single fully connected layer (~75K parameters)
  - **MLP Head**: Multi-layer perceptron with one hidden layer (~1M parameters)

#### Usage

**Inference with Linear Head:**
```bash
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_linear_head.pth \
    --head-type linear
```

**Inference with MLP Head:**
```bash
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_mlp_head.pth \
    --head-type mlp
```

#### Model Downloads

Model files are stored using Git LFS. If you encounter download issues, download directly from Google Drive:

- **DINOv2 Backbone Checkpoint** (`model_best.pth.tar`): [Download from Google Drive](https://drive.google.com/file/d/1FSI1YFiub6rrEfV9cruGdgWFBAyrbsm5/view?usp=sharing)
- **Linear Head** (`dino_linear_head.pth`): [Download from Google Drive](https://drive.google.com/file/d/1tNieT3O7WUsXRFA9r4bGndvzGnEhdjeL/view?usp=sharing)
- **MLP Head** (`dino_mlp_head.pth`): [Download from Google Drive](https://drive.google.com/file/d/1-F6FjfVUxikmlmU6zD0RxF5SXr6gMAVA/view?usp=sharing)

**Note**: After downloading, place files in:
- `model_best.pth.tar` → `DINOv2-linear-mlp/DINOv2/checkpoint/`
- `dino_linear_head.pth` → `DINOv2-linear-mlp/outputs/`
- `dino_mlp_head.pth` → `DINOv2-linear-mlp/outputs/`

For detailed documentation, see [`DINOv2-linear-mlp/README.md`](DINOv2-linear-mlp/README.md).

---

### 3. BioCLIP-2 LoRA Fine-tuning

**Location**: `bioclip-2/`

This project demonstrates **LoRA (Low-Rank Adaptation)** fine-tuning on BioCLIP-2, a state-of-the-art vision-language model for biological organism classification.

#### Requirements

- GPU with at least 16GB VRAM (tested on NVIDIA A100/V100)
- CUDA 11.7+
- ~50GB disk space for model weights and checkpoints

#### Installation

**Using conda:**
```bash
cd bioclip-2
conda env create -f requirements-training.yml
conda activate bioclip-train
```

**Using pip:**
```bash
cd bioclip-2
pip install -r requirements.txt
pip install torch>=2.0.1 torchvision>=0.15.2
pip install pandas pillow tqdm matplotlib
pip install huggingface-hub transformers
```

#### Quick Start

Run LoRA fine-tuning with your data:

```bash
cd bioclip-2
bash train_with_your_data.sh
```

#### Training Command

```bash
python -m src.training.main \
  --train-data /path/to/train_data.csv \
  --dataset-type csv \
  --csv-separator "\t" \
  --csv-img-key filepath \
  --csv-caption-key caption \
  --model ViT-L-14 \
  --batch-size 32 \
  --epochs 20 \
  --lr 5e-4 \
  --warmup 200 \
  --precision bf16 \
  --logs ./logs \
  --save-frequency 1 \
  --use-lora \
  --lora-rank 8 \
  --lora-alpha 16.0 \
  --lora-dropout 0.1 \
  --lora-enable-vision \
  --lora-enable-text
```

#### Evaluation

Evaluate the trained LoRA model:

```bash
python scripts/evaluate_lora_simple.py
```

#### Key Features

- **LoRA Parameters**: ~3M trainable parameters (vs ~430M total)
- **Checkpoint Size**: ~15MB per epoch (LoRA weights only)
- **Training Time**: ~9 minutes for 20 epochs on custom dataset

For detailed documentation, see [`bioclip-2/README.md`](bioclip-2/README.md).

---

### 4. Other Experimental Models

#### 4.1 BioClip-2 Original Model Inference

**Location**: `other_experiments/infer_bioclip.ipynb`

Inference using the original BioClip-2 model.

**Setup:**
1. Download `txt_emb_species.npy` and `txt_emb_species.json` from [imageomics/TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M/tree/main/embeddings)
2. Place them in the `other_experiments/` folder
3. Run the notebook: `other_experiments/infer_bioclip.ipynb`

#### 4.2 SigLIP2 + MLP Probing

**Location**: `sigLIP2_mlp/`

Training code: `sigLIP2_mlp/finetune_sigCLIP2.ipynb`

**Trained Checkpoints**: [Download from Google Drive](https://drive.google.com/file/d/1zuLxk9dsFKf10aijMNFi2PCHAs0uTcim/view?usp=sharing)

#### 4.3 SigLIP2 + LoRA

**Location**: `sigLIP2_lora/`

Training code: `sigLIP2_lora/lora.ipynb`

**Trained Checkpoints**: [Download from Google Drive](https://drive.google.com/file/d/1PZBBI0K0H1EyyTvjmD_ZscLOYllwp01I/view?usp=sharing)

---

## Project Structure

```
.
├── Task-aware Prediction Filtering/    # Task-aware prediction filtering
│   ├── Tile_inference.ipynb
│   ├── test_imgs/
│   ├── requirements.txt
│   └── species_ids.csv
├── DINOv2-linear-mlp/                  # DINOv2 classification
│   ├── infer_dinov2_head.py
│   ├── DINOv2/
│   ├── outputs/
│   └── examples/
├── bioclip-2/                          # BioCLIP-2 LoRA fine-tuning
│   ├── src/
│   ├── scripts/
│   ├── data/
│   └── train_with_your_data.sh
├── other_experiments/                  # BioClip-2 inference
│   └── infer_bioclip.ipynb
├── sigLIP2_mlp/                        # SigLIP2 + MLP
│   └── finetune_sigCLIP2.ipynb
└── sigLIP2_lora/                       # SigLIP2 + LoRA
    └── lora.ipynb
```

---

## Citation

### BioCLIP-2
```bibtex
@article{gu2025bioclip,
  title = {{B}io{CLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning}, 
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  eprint = {2505.23883},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
}
```

### LoRA
```bibtex
@article{hu2021lora,
  title = {LoRA: Low-Rank Adaptation of Large Language Models},
  author = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {arXiv preprint arXiv:2106.09685},
  year = {2021}
}
```

---

## License

This project is released under the MIT License.

---

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

### Model Downloads

The model files are stored using Git LFS in this repository. If you encounter issues downloading from GitHub (e.g., Git LFS quota exceeded), you can download the models directly from Google Drive:

- **DINOv2 Backbone Checkpoint** (`model_best.pth.tar`): [Download from Google Drive](https://drive.google.com/file/d/1FSI1YFiub6rrEfV9cruGdgWFBAyrbsm5/view?usp=sharing)
- **Linear Head** (`dino_linear_head.pth`): [Download from Google Drive](https://drive.google.com/file/d/1tNieT3O7WUsXRFA9r4bGndvzGnEhdjeL/view?usp=sharing)
- **MLP Head** (`dino_mlp_head.pth`): [Download from Google Drive](https://drive.google.com/file/d/1-F6FjfVUxikmlmU6zD0RxF5SXr6gMAVA/view?usp=sharing)

**Note**: After downloading from Google Drive, place the files in the corresponding directories:
- `model_best.pth.tar` → `DINOv2/checkpoint/`
- `dino_linear_head.pth` → `outputs/`
- `dino_mlp_head.pth` → `outputs/`

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

---

# BioCLIP 2 LoRA Fine-tuning Project

This project demonstrates **LoRA (Low-Rank Adaptation)** fine-tuning on [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2), a state-of-the-art vision-language model for biological organism classification. The implementation is based on [BioCLIP 2](https://github.com/imageomics/BioCLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Table of Contents

1. [Installation](#installation-1)
2. [Algorithm: LoRA Fine-tuning](#algorithm-lora-fine-tuning)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Project Structure](#project-structure-1)
8. [Citation](#citation-1)

---

## Installation

### Environment Setup

1. **Create conda environment:**

```bash
conda env create -f requirements-training.yml
```

Or install via pip:

```bash
pip install -r requirements.txt
```

2. **Additional dependencies for training:**

```bash
pip install torch>=2.0.1 torchvision>=0.15.2
pip install pandas pillow tqdm matplotlib
pip install huggingface-hub transformers
```

3. **Activate environment:**

```bash
conda activate bioclip-train
```

### Hardware Requirements

- GPU with at least 16GB VRAM (tested on NVIDIA A100/V100)
- CUDA 11.7+ supported
- ~50GB disk space for model weights and checkpoints

---

## Algorithm: LoRA Fine-tuning

### Overview

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This significantly reduces the number of trainable parameters while maintaining model performance.

### Implementation Details

The LoRA implementation is located in `src/open_clip/lora.py` and includes:

#### Core Components:

1. **LoRALayer** - Wraps linear layers with low-rank adaptation:
```python
# LoRA output: base_output + (x @ A.T @ B.T) * scale
# where scale = alpha / rank
```

2. **LoRAMultiheadAttention** - Applies LoRA to Q, K, V projections in Multi-Head Attention:
   - Separate LoRA matrices for Query, Key, and Value
   - Maintains original attention computation

3. **LoRAAttention** - Custom attention wrapper for CLIP's attention mechanism

#### Key Functions:

| Function | Description |
|----------|-------------|
| `apply_lora_to_model()` | Applies LoRA to CLIP model's vision and text encoders |
| `get_lora_parameters()` | Retrieves all trainable LoRA parameters |
| `save_lora_weights()` | Saves only LoRA weights to disk (~15MB vs ~900MB full model) |
| `load_lora_weights()` | Loads LoRA weights into model |

#### LoRA Hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rank` | 8 | Low-rank dimension |
| `alpha` | 16.0 | Scaling factor (scale = alpha/rank = 2.0) |
| `dropout` | 0.1 | LoRA dropout rate |
| `enable_vision` | True | Apply to vision encoder (ViT-L/14) |
| `enable_text` | True | Apply to text encoder |

### Model Architecture

```
CLIP with LoRA (ViT-L-14)
├── Visual Encoder (VisionTransformer)
│   ├── 24 ResidualAttentionBlocks
│   └── Each block: LoRAMultiheadAttention (embed_dim=1024)
└── Text Encoder (Transformer)
    ├── 12 ResidualAttentionBlocks
    └── Each block: LoRAMultiheadAttention (embed_dim=768)

Total LoRA modules applied: 36
Trainable parameters: ~3M (vs ~430M total)
```

---

## Dataset

### Data Structure

Training and validation data are stored in CSV format:

```
data/
├── train_data.csv      # Training set
├── val_data.csv        # Validation set (1853 samples)
└── annotation/         # Benchmark metadata
    ├── meta-album/     # Meta-Album datasets
    ├── nabirds/        # NABirds dataset
    └── rare_species/   # Rare Species dataset
```

### Data Format

CSV files use tab-separated format with two columns:

| Column | Description |
|--------|-------------|
| `filepath` | Absolute path to image file |
| `caption` | Class label or taxonomic ID |

Example:
```
filepath	caption
/path/to/image1.jpg	1355941
/path/to/image2.jpg	1355950
```

### Preparing Your Own Data

Use the provided script to prepare data from folder structure:

```bash
python prepare_dataset_from_folders.py --input-dir /path/to/images --output-dir ./data
```

---

## Training

### Quick Start

Run LoRA fine-tuning with your data:

```bash
bash train_with_your_data.sh
```

### Training Command

```bash
python -m src.training.main \
  --train-data /path/to/train_data.csv \
  --dataset-type csv \
  --csv-separator "\t" \
  --csv-img-key filepath \
  --csv-caption-key caption \
  --model ViT-L-14 \
  --batch-size 32 \
  --epochs 20 \
  --lr 5e-4 \
  --warmup 200 \
  --precision bf16 \
  --logs ./logs \
  --save-frequency 1 \
  --log-every-n-steps 10 \
  --workers 4 \
  --use-lora \
  --lora-rank 8 \
  --lora-alpha 16.0 \
  --lora-dropout 0.1 \
  --lora-enable-vision \
  --lora-enable-text
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model` | ViT-L-14 | CLIP model architecture |
| `--batch-size` | 32 | Batch size per GPU |
| `--epochs` | 20 | Total training epochs |
| `--lr` | 5e-4 | Learning rate |
| `--warmup` | 200 | Warmup steps |
| `--precision` | bf16 | Mixed precision training |

### Output

Training outputs are saved to `./logs/[timestamp]/`:

```
logs/
└── 2025_12_16-11_40_22-model_ViT-L-14-lr_0.0005-b_32-j_4-p_bf16/
    ├── checkpoints/
    │   ├── lora_epoch_1.pt
    │   ├── lora_epoch_2.pt
    │   └── ... (epoch 1-20)
    ├── out.log           # Training log
    ├── params.txt        # Training parameters
    └── tensorboard/      # TensorBoard events
```

---

## Evaluation

### Running Evaluation

Evaluate the trained LoRA model:

```bash
python scripts/evaluate_lora_simple.py
```

### Evaluation Script Configuration

Edit the script to set:
- `LORA_WEIGHTS_PATH`: Path to trained LoRA checkpoint
- `VAL_DATA_PATH`: Path to validation CSV

### Evaluation Process

1. Load base CLIP model (ViT-L-14)
2. Apply LoRA layers with same configuration as training
3. Load trained LoRA weights
4. Perform zero-shot classification on validation set
5. Compute Top-1 and Top-5 accuracy

### Visualization

Plot training loss curve:

```bash
python scripts/plot_loss_curve.py
```

Output: `logs/loss_curve.png`

---

## Results

### Training Results

Training on custom biological image dataset (5 classes, ~9000 training samples):

| Metric | Value |
|--------|-------|
| Training Epochs | 20 |
| Initial Loss | ~6.0 |
| Final Loss | ~2.09 |
| Loss Reduction | ~65% |
| Training Time | ~9 minutes |

### Loss Curve

![Training Loss Curve](logs/loss_curve.png)

The loss curve shows consistent convergence with cosine learning rate schedule.

### Checkpoints

All epoch checkpoints are saved:
- `lora_epoch_1.pt` to `lora_epoch_20.pt`
- Each checkpoint: ~15MB (LoRA weights only)

---

## Project Structure

```
bioclip-2/
├── src/
│   ├── open_clip/
│   │   ├── lora.py           # LoRA implementation
│   │   ├── model.py          # CLIP model
│   │   ├── factory.py        # Model factory
│   │   └── transformer.py    # Transformer modules
│   ├── training/
│   │   ├── main.py           # Training entry point
│   │   ├── train.py          # Training loop
│   │   └── data.py           # Data loading
│   └── evaluation/           # Evaluation scripts
├── scripts/
│   ├── evaluate_lora_simple.py
│   └── plot_loss_curve.py
├── data/
│   ├── train_data.csv
│   └── val_data.csv
├── logs/                     # Training outputs
├── train_with_your_data.sh   # Training script
├── requirements.txt
└── README.md
```

---

## Usage Examples

### 1. Load Trained LoRA Model

```python
import sys
sys.path.insert(0, 'src')

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.lora import apply_lora_to_model, load_lora_weights

# Create base model
model, _, preprocess = create_model_and_transforms('ViT-L-14', pretrained=None)

# Apply LoRA
model, _ = apply_lora_to_model(
    model, 
    rank=8, 
    alpha=16.0,
    enable_vision=True,
    enable_text=True
)

# Load trained weights
load_lora_weights(model, 'logs/.../checkpoints/lora_epoch_20.pt')
model = model.cuda().eval()
```

### 2. Inference

```python
from PIL import Image

# Load and preprocess image
image = Image.open('test_image.jpg').convert('RGB')
image_input = preprocess(image).unsqueeze(0).cuda()

# Get tokenizer
tokenizer = get_tokenizer('ViT-L-14')

# Create text features
texts = ['a photo of species A', 'a photo of species B']
text_tokens = tokenizer(texts).cuda()

# Compute features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)
    
    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(f"Predictions: {similarity}")
```

---

## Citation

If you use this code, please cite:

### BioCLIP 2
```bibtex
@article{gu2025bioclip,
  title = {{B}io{CLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning}, 
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  eprint = {2505.23883},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
}
```

### LoRA
```bibtex
@article{hu2021lora,
  title = {LoRA: Low-Rank Adaptation of Large Language Models},
  author = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {arXiv preprint arXiv:2106.09685},
  year = {2021}
}
```

### OpenCLIP
```bibtex
@software{ilharco_gabriel_2021_5143773,
  author = {Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title = {OpenCLIP},
  year = {2021},
  doi = {10.5281/zenodo.5143773},
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
