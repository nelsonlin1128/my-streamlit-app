# resize_512.py
import os
from PIL import Image

src_dir = "data/styles/jojo"
dst_dir = "data/styles/jojo_512"
os.makedirs(dst_dir, exist_ok=True)

for fn in os.listdir(src_dir):
    if not fn.lower().endswith((".jpg",".png")):
        continue
    im = Image.open(os.path.join(src_dir, fn)).convert("RGB")
    im = im.resize((512,512), Image.LANCZOS)
    im.save(os.path.join(dst_dir, fn), quality=95)
print("All images resized to 512×512 in", dst_dir)
