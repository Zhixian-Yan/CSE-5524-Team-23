# BioCLIP 2 LoRA Fine-tuning Project

This project demonstrates **LoRA (Low-Rank Adaptation)** fine-tuning on [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2), a state-of-the-art vision-language model for biological organism classification. The implementation is based on [BioCLIP 2](https://github.com/imageomics/BioCLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Table of Contents

1. [Installation](#installation)
2. [Algorithm: LoRA Fine-tuning](#algorithm-lora-fine-tuning)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Citation](#citation)

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
