#!/usr/bin/env python3
"""
简化版评估脚本：对比随机初始化模型和 LoRA 微调后的模型
不需要下载外部预训练模型
"""

import os
import sys
sys.path.insert(0, '/root/bioclip-2/src')

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.lora import apply_lora_to_model, load_lora_weights

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-L-14"
LORA_WEIGHTS_PATH = "/root/bioclip-2/logs/2025_12_16-11_40_22-model_ViT-L-14-lr_0.0005-b_32-j_4-p_bf16/checkpoints/lora_epoch_20.pt"
VAL_DATA_PATH = "/root/bioclip-2/data/val_data.csv"

# LoRA 配置（需要与训练时一致）
LORA_RANK = 8
LORA_ALPHA = 16.0


def load_lora_model():
    """加载带有 LoRA 权重的模型"""
    print("加载 LoRA 微调后的模型...")
    model, _, preprocess = create_model_and_transforms(
        MODEL_NAME,
        pretrained=None  # 与训练时一致
    )
    
    # 应用 LoRA
    model, _ = apply_lora_to_model(
        model, 
        rank=LORA_RANK, 
        alpha=LORA_ALPHA,
        enable_vision=True,
        enable_text=True
    )
    
    # 加载 LoRA 权重
    load_lora_weights(model, LORA_WEIGHTS_PATH)
    model = model.to(DEVICE).eval()
    return model, preprocess


def load_random_baseline():
    """加载随机初始化模型（训练起点参考）"""
    print("加载随机初始化模型 (训练起点参考)...")
    model, _, preprocess = create_model_and_transforms(
        MODEL_NAME,
        pretrained=None
    )
    model = model.to(DEVICE).eval()
    return model, preprocess


def get_class_names(df):
    """获取类别名称列表"""
    return sorted(df['caption'].unique().tolist())


def create_text_features(model, tokenizer, class_names):
    """为所有类别创建文本特征"""
    templates = [
        "a photo of {}",
        "an image of {}",
        "{}"
    ]
    
    text_features_list = []
    
    with torch.no_grad():
        for class_name in class_names:
            class_features = []
            for template in templates:
                text = template.format(class_name)
                tokens = tokenizer([text]).to(DEVICE)
                features = model.encode_text(tokens)
                # 处理可能返回 tuple 的情况
                if isinstance(features, tuple):
                    features = features[0]
                features = features / features.norm(dim=-1, keepdim=True)
                class_features.append(features)
            
            avg_features = torch.stack(class_features).mean(dim=0)
            avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
            text_features_list.append(avg_features)
    
    text_features = torch.cat(text_features_list, dim=0)
    return text_features


def evaluate_model(model, preprocess, tokenizer, df, class_names, model_name="Model"):
    """评估模型在验证集上的性能"""
    print(f"\n评估 {model_name}...")
    
    text_features = create_text_features(model, tokenizer, class_names)
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"评估 {model_name}"):
            filepath = row['filepath']
            true_label = str(row['caption'])
            
            try:
                image = Image.open(filepath).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(DEVICE)
                
                image_features = model.encode_image(image_input)
                # 处理可能返回 tuple 的情况
                if isinstance(image_features, tuple):
                    image_features = image_features[0]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                values, indices = similarity[0].topk(5)
                top1_pred = class_names[indices[0].item()]
                top5_preds = [class_names[i.item()] for i in indices]
                
                if top1_pred == true_label:
                    correct_top1 += 1
                    class_correct[true_label] += 1
                
                if true_label in top5_preds:
                    correct_top5 += 1
                
                class_total[true_label] += 1
                total += 1
                
            except Exception as e:
                print(f"处理 {filepath} 时出错: {e}")
                continue
    
    top1_acc = correct_top1 / total * 100 if total > 0 else 0
    top5_acc = correct_top5 / total * 100 if total > 0 else 0
    
    print(f"\n{model_name} 结果:")
    print(f"  Top-1 准确率: {top1_acc:.2f}%")
    print(f"  Top-5 准确率: {top5_acc:.2f}%")
    print(f"  总样本数: {total}")
    
    print(f"\n  各类别 Top-1 准确率:")
    for class_name in class_names:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name] * 100
            print(f"    {class_name}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
    
    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'total': total,
        'class_acc': {c: class_correct[c]/class_total[c]*100 if class_total[c] > 0 else 0 for c in class_names}
    }


def main():
    print("="*60)
    print("LoRA 微调效果验证评估")
    print("="*60)
    
    print(f"\n加载验证数据: {VAL_DATA_PATH}")
    df = pd.read_csv(VAL_DATA_PATH, sep='\t')
    print(f"验证集样本数: {len(df)}")
    
    class_names = get_class_names(df)
    print(f"类别数: {len(class_names)}")
    print(f"类别: {class_names}")
    
    tokenizer = get_tokenizer(MODEL_NAME)
    
    results = {}
    
    # 1. 评估随机初始化模型（训练起点）
    print("\n" + "-"*40)
    random_model, preprocess = load_random_baseline()
    results['random_init'] = evaluate_model(
        random_model, preprocess, tokenizer, df, class_names,
        model_name="随机初始化模型 (训练起点)"
    )
    del random_model
    torch.cuda.empty_cache()
    
    # 2. 评估 LoRA 微调后的模型
    print("\n" + "-"*40)
    lora_model, preprocess = load_lora_model()
    results['lora'] = evaluate_model(
        lora_model, preprocess, tokenizer, df, class_names,
        model_name="LoRA 微调模型 (20 epochs)"
    )
    
    # 对比结果
    print("\n" + "="*60)
    print("📊 对比结果总结")
    print("="*60)
    
    print(f"\n{'模型':<35} {'Top-1 准确率':<15} {'Top-5 准确率':<15}")
    print("-"*65)
    print(f"{'随机初始化 (训练起点)':<35} {results['random_init']['top1_acc']:<15.2f} {results['random_init']['top5_acc']:<15.2f}")
    print(f"{'LoRA微调后 (20 epochs)':<35} {results['lora']['top1_acc']:<15.2f} {results['lora']['top5_acc']:<15.2f}")
    
    # 计算提升
    improvement_top1 = results['lora']['top1_acc'] - results['random_init']['top1_acc']
    improvement_top5 = results['lora']['top5_acc'] - results['random_init']['top5_acc']
    
    print(f"\n📈 LoRA 微调效果:")
    print(f"   Top-1 提升: {'+' if improvement_top1 >= 0 else ''}{improvement_top1:.2f}%")
    print(f"   Top-5 提升: {'+' if improvement_top5 >= 0 else ''}{improvement_top5:.2f}%")
    
    # 各类别对比
    print(f"\n📋 各类别对比:")
    for class_name in class_names:
        random_acc = results['random_init']['class_acc'][class_name]
        lora_acc = results['lora']['class_acc'][class_name]
        diff = lora_acc - random_acc
        print(f"   {class_name}: {random_acc:.1f}% → {lora_acc:.1f}% ({'+' if diff >= 0 else ''}{diff:.1f}%)")
    
    # 保存结果
    results_path = "/root/bioclip-2/logs/evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write("LoRA 微调效果验证评估结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"验证集样本数: {len(df)}\n")
        f.write(f"类别数: {len(class_names)}\n")
        f.write(f"类别: {class_names}\n\n")
        
        f.write(f"1. 随机初始化模型 (训练起点):\n")
        f.write(f"   Top-1 准确率: {results['random_init']['top1_acc']:.2f}%\n")
        f.write(f"   Top-5 准确率: {results['random_init']['top5_acc']:.2f}%\n\n")
        
        f.write(f"2. LoRA 微调模型 (20 epochs):\n")
        f.write(f"   Top-1 准确率: {results['lora']['top1_acc']:.2f}%\n")
        f.write(f"   Top-5 准确率: {results['lora']['top5_acc']:.2f}%\n\n")
        
        f.write(f"📈 提升效果:\n")
        f.write(f"   Top-1: {'+' if improvement_top1 >= 0 else ''}{improvement_top1:.2f}%\n")
        f.write(f"   Top-5: {'+' if improvement_top5 >= 0 else ''}{improvement_top5:.2f}%\n\n")
        
        f.write(f"各类别对比:\n")
        for class_name in class_names:
            random_acc = results['random_init']['class_acc'][class_name]
            lora_acc = results['lora']['class_acc'][class_name]
            diff = lora_acc - random_acc
            f.write(f"   {class_name}: {random_acc:.1f}% → {lora_acc:.1f}% ({'+' if diff >= 0 else ''}{diff:.1f}%)\n")
    
    print(f"\n✅ 结果已保存到: {results_path}")
    
    return results


if __name__ == "__main__":
    main()

