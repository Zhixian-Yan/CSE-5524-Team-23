# Final Project – PlantCLEF 2025 Team-23

## Installation
- Python 3.11
- pip install -r requirements.txt

## Pretrained Model
Download the pretrained model <https://drive.google.com/file/d/15Yxi9vovUxo4YUMWYcO2k3JqzFPQgEBm/view?usp=sharing> and PlantCLEF2024_single_plant_training_metadata.csv <https://drive.google.com/file/d/1z3gx4W6Vj9iK0V-o5pS96rcIZeBShuPX/view?usp=sharing> 
save them to this folder.

## Test Examples
We provide several validation images under:
<test_imgs>

## Run in jupyter notebook
Tile_inference.ipynb to see the plantCLEF 2025 quandrat prediction

## Other model trained:
### 1. BioCLIP-2 + LoRA Fine-tuning
LoRA (Low-Rank Adaptation) fine-tuning on [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) for biological organism classification.

**Key Features:**
- Trainable parameters: ~3M (vs ~430M total)
- LoRA rank: 8, alpha: 16.0
- Applied to both vision (ViT-L/14) and text encoders

**Quick Start:**
```bash
cd bioclip-2
conda env create -f requirements-training.yml
conda activate bioclip-train
bash train_with_your_data.sh
```

**Training Results:** Initial Loss ~6.0 → Final Loss ~2.09 (65% reduction)

For detailed documentation, see [bioclip-2/README.md](bioclip-2/README.md)

### 2. SigLIP2 + MLP probing
 training code: sigLIP2_mlp/finetune_sigCLIP2.ipynb
 trained checkpoints: <https://drive.google.com/file/d/1zuLxk9dsFKf10aijMNFi2PCHAs0uTcim/view?usp=sharing>
### 3. SigLIP2 + LORA
 training code: sigLIP2_lora/lora.ipynb
 trained checkpoints: https://drive.google.com/file/d/1PZBBI0K0H1EyyTvjmD_ZscLOYllwp01I/view?usp=sharing

 
