# Final Project – PlantCLEF 2025 Team-23

### 1. Final method: Task-aware Prediction Filtering

 **Location**: `Task-aware Prediction Filtering/` 
 
**Installation:** 

- Python 3.11
- pip install -r requirements.txt

**Pretrained Model:** 

Download the pretrained model <https://drive.google.com/file/d/15Yxi9vovUxo4YUMWYcO2k3JqzFPQgEBm/view?usp=sharing> and 

PlantCLEF2024_single_plant_training_metadata.csv <https://drive.google.com/file/d/1z3gx4W6Vj9iK0V-o5pS96rcIZeBShuPX/view?usp=sharing> 
save them to folder `Task-aware Prediction Filtering/`.

**Test Examples:**  

We provide several validation images under:
`Task-aware Prediction Filtering/test_imgs`

**Run in jupyter notebook:**   

Tile_inference.ipynb to see the plantCLEF 2025 quandrat prediction

run enviroments: Linux (Ubuntu 24.04) nvidia-driver 580 cuda 13.0

## Other model trained:
### 2. BioCLIP-2 + LoRA Fine-tuning

**Location**: `bioclip-2/`

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

### 3. DINOv2 + Linear/MLP Classification

**Location**: `DINOv2-linear-mlp/`

Frozen DINOv2 backbone with lightweight classification heads (Linear and MLP) for plant species classification.

**Key Features:**
- **Backbone**: DINOv2 ViT-B/14 (frozen, pretrained)
- **Linear Head**: ~75K trainable parameters
- **MLP Head**: ~1M trainable parameters
- **Training**: 5 epochs, Cross-Entropy Loss, Adam optimizer

**Quick Start:**
```bash
cd DINOv2-linear-mlp
pip install -r requirements.txt
python infer_dinov2_head.py \
    --img examples/example_1.jpg \
    --species-ids DINOv2/dataset/species_ids.csv \
    --checkpoint DINOv2/checkpoint/model_best.pth.tar \
    --head-pth outputs/dino_linear_head.pth \
    --head-type linear
```

**Model Downloads:**
- [DINOv2 Backbone Checkpoint](https://drive.google.com/file/d/1FSI1YFiub6rrEfV9cruGdgWFBAyrbsm5/view?usp=sharing)
- [Linear Head](https://drive.google.com/file/d/1tNieT3O7WUsXRFA9r4bGndvzGnEhdjeL/view?usp=sharing)
- [MLP Head](https://drive.google.com/file/d/1-F6FjfVUxikmlmU6zD0RxF5SXr6gMAVA/view?usp=sharing)

For detailed documentation, see [`DINOv2-linear-mlp/README.md`](DINOv2-linear-mlp/README.md).

### 4. SigLIP2 + MLP probing

**Location**: `sigLIP2_mlp/`

 training code: sigLIP2_mlp/finetune_sigCLIP2.ipynb
 trained checkpoints: <https://drive.google.com/file/d/1zuLxk9dsFKf10aijMNFi2PCHAs0uTcim/view?usp=sharing>
### 5. SigLIP2 + LORA

**Location**: `sigLIP2_lora/`

 training code: sigLIP2_lora/lora.ipynb
 trained checkpoints: https://drive.google.com/file/d/1PZBBI0K0H1EyyTvjmD_ZscLOYllwp01I/view?usp=sharing

 
