# collect_ghibli.py
import os
import shutil
from datasets import load_dataset
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi
from icrawler.builtin import GoogleImageCrawler

BASE_DIR = "data/styles/ghibli"
os.makedirs(BASE_DIR, exist_ok=True)

# 1. 从 Hugging Face 下载 Nechintosh/ghibli
print("1. Downloading Hugging Face Nechintosh/ghibli dataset…")
ds = load_dataset("Nechintosh/ghibli", split="train")
for idx, item in enumerate(ds):
    img = item["image"]  # PIL.Image.Image
    img.save(os.path.join(BASE_DIR, f"hf_{idx:04d}.jpg"))
print(f"   → Saved {len(ds)} images from Hugging Face.")

# 2. 从 Kaggle 下载并自动合并
print("2. Downloading Kaggle real-to-ghibli dataset…")
api = KaggleApi(); api.authenticate()
api.dataset_download_files(
    "shubham1921/real-to-ghibli-image-dataset-5k-paired-images",
    path=BASE_DIR,
    unzip=True
)

# 自动探测解压后的目录
for name in os.listdir(BASE_DIR):
    candidate = os.path.join(BASE_DIR, name)
    if os.path.isdir(candidate) and "real-to-ghibli-image-dataset" in name:
        # 判断里面是不是直接存了图片，还是还有一层子文件夹
        # 如果有 images 子文件夹，就用它，否则直接用 candidate
        inner = os.path.join(candidate, "images")
        source_dir = inner if os.path.isdir(inner) else candidate

        # 把所有文件搬到 BASE_DIR，并加上前缀
        for fn in os.listdir(source_dir):
            src = os.path.join(source_dir, fn)
            dst = os.path.join(BASE_DIR, f"kaggle_{fn}")
            shutil.move(src, dst)

        # 删除解压出来的多余文件夹
        shutil.rmtree(candidate)
        break

print("   → Merged images from Kaggle into", BASE_DIR)

# 3. 用 icrawler 从 Google Images 抓取额外截图（500 张）
print("3. Crawling Google Images for additional Ghibli screenshots…")
extra_dir = os.path.join(BASE_DIR, "google")
os.makedirs(extra_dir, exist_ok=True)
crawler = GoogleImageCrawler(storage={"root_dir": extra_dir})
crawler.crawl(
    keyword="Studio Ghibli anime screenshot",
    max_num=500,
    min_size=(400, 400)
)
print("   → Saved 500 images via Google Search.")
