#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip2-lora
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

##### LoRA Fine-tuning Script for BioCLIP 2
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# Set paths - UPDATE THESE TO YOUR DATA PATHS
TRAIN_DATA="[your-train-data-path]"
VAL_DATA="[your-val-data-path]"
PRETRAINED_MODEL="imageomics/bioclip-2"  # or path to your BioCLIP 2 checkpoint

# LoRA parameters
LORA_RANK=8
LORA_ALPHA=16.0
LORA_DROPOUT=0.1

# Training parameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
EPOCHS=10
WARMUP=100

python -m src.training.main \
  --train-data "${TRAIN_DATA}" \
  --val-data "${VAL_DATA}" \
  --dataset-type 'webdataset' \
  --pretrained "${PRETRAINED_MODEL}" \
  --model ViT-L-14 \
  --batch-size ${BATCH_SIZE} \
  --accum-freq 1 \
  --epochs ${EPOCHS} \
  --workers 4 \
  --lr ${LEARNING_RATE} \
  --warmup ${WARMUP} \
  --seed 42 \
  --precision bf16 \
  --logs './logs' \
  --save-frequency 1 \
  --log-every-n-steps 10 \
  --use-lora \
  --lora-rank ${LORA_RANK} \
  --lora-alpha ${LORA_ALPHA} \
  --lora-dropout ${LORA_DROPOUT} \
  --lora-enable-vision \
  --lora-enable-text


