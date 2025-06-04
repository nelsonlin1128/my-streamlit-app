import os
from datasets import load_dataset
from google_images_download import google_images_download
from PIL import Image

# 确保目标目录存在
os.makedirs('data/styles/jojo', exist_ok=True)

# 1. 从 Hugging Face 下载 Stone Ocean JoJo 数据集（1376 张左右）
ds = load_dataset('norod78/jojo-stone-ocean-blip-captions-512', split='train')
for idx, item in enumerate(ds):
    img = item['image'].convert("RGB")
    img.save(f'data/styles/jojo/hf_{idx:04d}.jpg')
print(f"已下载 {len(ds)} 张 Hugging Face JoJo 图像")


# 2. 使用 google_images_download 批量下载 JoJo 动漫截图（示例 500 张）
response = google_images_download.googleimagesdownload()
arguments = {
    'keywords': "JoJo's Bizarre Adventure anime screenshot",
    'limit': 700,
    'print_urls': False,
    'output_directory': 'data/styles/jojo',
    'format': 'jpg'
}
paths = response.download(arguments)
print("google_images_download 下载路径：", paths)
