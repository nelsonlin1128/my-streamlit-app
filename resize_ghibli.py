# resize_ghibli.py
import os
from PIL import Image

SRC_DIR = "data/styles/ghibli"
DST_DIR = "data/styles/ghibli_512"

for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        if not fn.lower().endswith((".jpg", ".png")):
            continue
        src_path = os.path.join(root, fn)
        # 计算相对路径，在 DST_DIR 下重建相同子目录
        rel_path = os.path.relpath(src_path, SRC_DIR)
        dst_path = os.path.join(DST_DIR, rel_path)

        # 确保目标子目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 打开、RGB 转换、Resize、保存
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(dst_path, quality=95)

print("✅ 所有 Ghibli 图片已统一为 512×512，保存在", DST_DIR)
