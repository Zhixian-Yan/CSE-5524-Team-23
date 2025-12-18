#!/usr/bin/env python3
"""
从训练日志中提取Loss并绘制曲线
"""

import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 日志文件路径
log_path = "/root/bioclip-2/logs/2025_12_16-11_40_22-model_ViT-L-14-lr_0.0005-b_32-j_4-p_bf16/out.log"

epochs = []
losses = []
lrs = []

print("解析训练日志...")

with open(log_path, 'r') as f:
    for line in f:
        # 匹配 epoch 结束时的 loss（100%的那一行）
        match = re.search(r'Train Epoch: (\d+) \[.*\(100%\)\].*Loss: [\d.]+ \(([\d.]+)\)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)
            
        # 提取学习率
        lr_match = re.search(r'LR: ([\d.]+)', line)
        if lr_match and '100%' in line:
            lrs.append(float(lr_match.group(1)))

print(f"找到 {len(epochs)} 个epoch的数据")

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 绘制 Loss 曲线
ax1 = axes[0]
ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6, label='Contrastive Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('LoRA Fine-tuning Training Loss', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 添加起止点标注
ax1.annotate(f'Start: {losses[0]:.3f}', xy=(epochs[0], losses[0]), 
             xytext=(epochs[0]+1, losses[0]+0.1),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, color='green')
ax1.annotate(f'End: {losses[-1]:.3f}', xy=(epochs[-1], losses[-1]), 
             xytext=(epochs[-1]-3, losses[-1]+0.1),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, color='red')

# 绘制学习率曲线
if lrs:
    ax2 = axes[1]
    ax2.plot(epochs[:len(lrs)], lrs, 'r-s', linewidth=2, markersize=4, label='Learning Rate')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule (Cosine)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()

# 保存图表
output_path = '/root/bioclip-2/logs/loss_curve.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存到: {output_path}")

# 打印统计信息
print("\n" + "="*50)
print("训练统计")
print("="*50)
print(f"初始 Loss: {losses[0]:.4f}")
print(f"最终 Loss: {losses[-1]:.4f}")
print(f"Loss 下降: {(1 - losses[-1]/losses[0])*100:.1f}%")
print(f"最小 Loss: {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})")
print(f"总 Epochs: {len(epochs)}")

