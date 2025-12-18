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
### 1. infer using original BioClip-2 model
 download txt_emb_species.npy and txt_emb_species.json (from imageomics/TreeOfLife-200M) <https://huggingface.co/datasets/imageomics/TreeOfLife-200M/tree/main/embeddings>
 put them in the same folder.
 code: other_experiments/infer_bioclip.ipynb
### 2. SigLIP2 + MLP probing
 training code: sigLIP2_mlp/finetune_sigCLIP2.ipynb
 trained checkpoints: <https://drive.google.com/file/d/1zuLxk9dsFKf10aijMNFi2PCHAs0uTcim/view?usp=sharing>
### 3. SigLIP2 + LORA
 training code: sigLIP2_lora/lora.ipynb
 trained checkpoints: https://drive.google.com/file/d/1PZBBI0K0H1EyyTvjmD_ZscLOYllwp01I/view?usp=sharing

 
