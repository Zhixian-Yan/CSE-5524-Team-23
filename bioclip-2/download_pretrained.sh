#!/bin/bash
# 手动下载预训练权重的脚本

echo "正在下载预训练权重..."

# 创建缓存目录
mkdir -p ~/.cache/clip

# 尝试下载 laion400m_e31（GitHub链接，可能更稳定）
echo "尝试从GitHub下载 laion400m_e31..."
wget --timeout=30 --tries=3 \
  https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt \
  -O ~/.cache/clip/vit_l_14-laion400m_e31-69988bb6.pt

if [ $? -eq 0 ]; then
    echo "✅ 下载成功！"
    echo "权重保存在: ~/.cache/clip/vit_l_14-laion400m_e31-69988bb6.pt"
    echo ""
    echo "现在可以使用以下命令训练："
    echo "  --pretrained ~/.cache/clip/vit_l_14-laion400m_e31-69988bb6.pt"
else
    echo "❌ 下载失败，请检查网络连接"
    echo ""
    echo "您可以："
    echo "1. 检查网络连接"
    echo "2. 使用代理"
    echo "3. 从其他机器下载后上传到服务器"
fi

