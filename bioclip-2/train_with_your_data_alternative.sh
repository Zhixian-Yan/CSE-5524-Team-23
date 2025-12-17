#!/bin/bash
# LoRA Fine-tuning训练脚本 - 使用备用预训练权重（laion400m_e31，有直接下载链接）

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bioclip-train

cd /root/bioclip-2

python -m src.training.main \
  --train-data /root/bioclip-2/data/train_data.csv \
  --dataset-type csv \
  --csv-separator "\t" \
  --csv-img-key filepath \
  --csv-caption-key caption \
  --pretrained laion400m_e31 \
  --model ViT-L-14 \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-4 \
  --warmup 100 \
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

