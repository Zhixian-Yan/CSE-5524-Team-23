#!/bin/bash
# 继续训练脚本 - 从 LoRA 权重恢复，训练剩余 epoch

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bioclip-train

cd /root/bioclip-2

# 从 epoch 14 的 LoRA 权重加载，继续训练到 epoch 20
python -m src.training.main \
  --train-data /root/bioclip-2/data/train_data.csv \
  --dataset-type csv \
  --csv-separator "\t" \
  --csv-img-key filepath \
  --csv-caption-key caption \
  --model ViT-L-14 \
  --batch-size 32 \
  --epochs 6 \
  --lr 1e-4 \
  --warmup 50 \
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
  --lora-enable-text \
  --lora-weights /root/bioclip-2/logs/2025_12_15-12_08_56-model_ViT-L-14-lr_0.0005-b_32-j_4-p_bf16/checkpoints/lora_epoch_14.pt
