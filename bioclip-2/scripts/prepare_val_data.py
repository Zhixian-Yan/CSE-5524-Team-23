#!/usr/bin/env python3
"""
创建验证数据集CSV文件
使用新加入的5个文件夹：1355941, 1355950, 1355977, 1356023, 1356040
"""

import os
import pandas as pd

# 验证数据文件夹（新加入的5个）
val_folders = ['1355941', '1355950', '1355977', '1356023', '1356040']
base_path = '/root/autodl-tmp/.autodl'

data = []

for folder in val_folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(folder_path, img_file)
                # caption 使用文件夹名（物种ID）
                caption = folder
                data.append({'filepath': filepath, 'caption': caption})
        print(f"文件夹 {folder}: {len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])} 张图片")
    else:
        print(f"警告: 文件夹 {folder_path} 不存在")

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存到CSV
output_path = '/root/bioclip-2/data/val_data.csv'
df.to_csv(output_path, sep='\t', index=False)

print(f"\n验证数据集已保存到: {output_path}")
print(f"总样本数: {len(df)}")
print(f"类别数: {df['caption'].nunique()}")
print(f"\n各类别样本数:")
print(df['caption'].value_counts())

