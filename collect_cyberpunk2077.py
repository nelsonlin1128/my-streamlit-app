# collect_cyberpunk2077.py

import os
from kaggle.api.kaggle_api_extended import KaggleApi
from icrawler.builtin import GoogleImageCrawler

# 1. 从 Kaggle 下载 CyberVerse 赛博朋克图集
api = KaggleApi()
api.authenticate()
dataset = "cyanex1702/cyberversecyberpunk-imagesdataset"
dest_dir = "data/styles/cyberpunk2077/kaggle"
os.makedirs(dest_dir, exist_ok=True)
print(f"下载 Kaggle 数据集 {dataset} 到 {dest_dir} …")
api.dataset_download_files(dataset, path=dest_dir, unzip=True)

# 2. 用 icrawler 补充额外截图（示例：500 张）
extra_dir = "data/styles/cyberpunk2077/google"
os.makedirs(extra_dir, exist_ok=True)
crawler = GoogleImageCrawler(storage={'root_dir': extra_dir})
print("用 icrawler 从 Google Images 下载 “Cyberpunk 2077 screenshot” …")
crawler.crawl(
    keyword="Cyberpunk 2077 screenshot",
    max_num=500,
    min_size=(400, 400),
    max_size=None
)

print("采集完成！目录结构：")
print("\n".join(os.listdir("data/styles/cyberpunk2077")))
