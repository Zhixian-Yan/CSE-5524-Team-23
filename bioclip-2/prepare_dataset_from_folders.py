#!/usr/bin/env python3
"""
将文件夹结构的植物图片数据集转换为CSV格式，用于BioCLIP 2训练

使用方法:
    python prepare_dataset_from_folders.py \
        --input-dir /path/to/your/train/folder \
        --output-csv train_data.csv \
        --caption-strategy folder_name

数据格式:
    输入: 
        train/
          1355871/
            image1.jpg
            image2.jpg
          ...
    
    输出: CSV文件，包含 filepath 和 caption 列
"""
import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}


def get_image_files(folder_path):
    """获取文件夹中的所有图片文件"""
    image_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'**/*{ext}'))  # 递归搜索
    
    return sorted(image_files)


def generate_caption_from_folder_name(folder_name, strategy='folder_name'):
    """
    根据文件夹名称生成文本描述
    
    strategy选项:
    - 'folder_name': 直接使用文件夹名作为描述
    - 'prefix': 添加前缀，如 "This is a plant species: {folder_name}"
    - 'number': 如果文件夹名是数字，可以转换为描述
    """
    if strategy == 'folder_name':
        return str(folder_name)
    elif strategy == 'prefix':
        return f"This is a plant species: {folder_name}"
    elif strategy == 'number':
        # 如果文件夹名是数字ID，可以生成描述
        if folder_name.isdigit():
            return f"Plant species ID: {folder_name}"
        else:
            return str(folder_name)
    else:
        return str(folder_name)


def create_csv_from_folders(input_dir, output_csv, caption_strategy='folder_name', 
                           use_absolute_path=True, max_samples=None):
    """
    从文件夹结构创建CSV文件
    
    Args:
        input_dir: 输入文件夹路径（包含多个子文件夹，每个子文件夹是一个类别）
        output_csv: 输出CSV文件路径
        caption_strategy: 生成文本描述的策略
        use_absolute_path: 是否使用绝对路径
        max_samples: 最大样本数（用于测试，None表示使用所有）
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"输入文件夹不存在: {input_dir}")
    
    print(f"正在扫描文件夹: {input_dir}")
    
    # 收集所有图片
    all_images = []
    all_captions = []
    
    # 获取所有子文件夹
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        # 如果没有子文件夹，直接在当前文件夹中查找图片
        print("未找到子文件夹，在当前文件夹中查找图片...")
        images = get_image_files(input_path)
        for img_path in tqdm(images, desc="处理图片"):
            if use_absolute_path:
                all_images.append(str(img_path.absolute()))
            else:
                all_images.append(str(img_path))
            # 使用文件夹名或文件名作为描述
            folder_name = input_path.name
            caption = generate_caption_from_folder_name(folder_name, caption_strategy)
            all_captions.append(caption)
    else:
        # 遍历每个子文件夹
        for folder in tqdm(subfolders, desc="处理文件夹"):
            folder_name = folder.name
            images = get_image_files(folder)
            
            if not images:
                print(f"警告: 文件夹 {folder_name} 中没有找到图片")
                continue
            
            # 生成该文件夹的文本描述
            caption = generate_caption_from_folder_name(folder_name, caption_strategy)
            
            # 添加所有图片
            for img_path in images:
                if use_absolute_path:
                    all_images.append(str(img_path.absolute()))
                else:
                    # 使用相对路径
                    all_images.append(str(img_path.relative_to(input_path.parent)))
                all_captions.append(caption)
    
    if not all_images:
        raise ValueError("没有找到任何图片文件！")
    
    # 限制样本数（用于测试）
    if max_samples and len(all_images) > max_samples:
        print(f"限制样本数从 {len(all_images)} 到 {max_samples}")
        all_images = all_images[:max_samples]
        all_captions = all_captions[:max_samples]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'filepath': all_images,
        'caption': all_captions
    })
    
    # 保存CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ 成功创建CSV文件: {output_csv}")
    print(f"   总图片数: {len(df)}")
    print(f"   唯一类别数: {df['caption'].nunique()}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 显示一些统计信息
    print("\n前5个样本:")
    print(df.head().to_string())
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='将文件夹结构的图片数据集转换为CSV格式'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='输入文件夹路径（包含多个子文件夹，每个子文件夹是一个类别）'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        required=True,
        help='输出CSV文件路径'
    )
    parser.add_argument(
        '--caption-strategy',
        type=str,
        default='folder_name',
        choices=['folder_name', 'prefix', 'number'],
        help='生成文本描述的策略: folder_name(使用文件夹名), prefix(添加前缀), number(数字ID)'
    )
    parser.add_argument(
        '--relative-path',
        action='store_true',
        help='使用相对路径而不是绝对路径'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大样本数（用于测试，None表示使用所有）'
    )
    
    args = parser.parse_args()
    
    try:
        create_csv_from_folders(
            input_dir=args.input_dir,
            output_csv=args.output_csv,
            caption_strategy=args.caption_strategy,
            use_absolute_path=not args.relative_path,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


