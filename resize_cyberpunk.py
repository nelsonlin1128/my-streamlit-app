# resize_cyberpunk.py
import os
from PIL import Image

# 原始目录
SRC_DIR = "data/styles/cyberpunk2077"
# 输出目录
DST_DIR = "data/styles/cyberpunk2077_512"

for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        if not fn.lower().endswith((".jpg", ".png")):
            continue
        src_path = os.path.join(root, fn)
        # 计算相对路径，用来在 DST_DIR 下重建目录结构
        rel_path = os.path.relpath(src_path, SRC_DIR)
        dst_path = os.path.join(DST_DIR, rel_path)

        # 确保目标子目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 打开、转换、缩放、保存
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(dst_path, quality=95)

print("所有 Cyberpunk2077 图片已统一为 512×512，保存在", DST_DIR)
