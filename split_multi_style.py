# split_multi_style.py
import os, random, shutil

random.seed(42)
BASE = "data/styles"
DST = "data/multi_style"  # 新根目录
STYLES = ["jojo_512", "cyberpunk2077_512", "ghibli_512"]
TRAIN_RATIO = 0.9

for style in STYLES:
    src_dir = os.path.join(BASE, style)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]
    random.shuffle(files)
    n_train = int(len(files) * TRAIN_RATIO)

    for phase, subset in [("train", files[:n_train]), ("val", files[n_train:])]:
        out_dir = os.path.join(DST, phase, style)
        os.makedirs(out_dir, exist_ok=True)
        for fn in subset:
            shutil.copy(os.path.join(src_dir, fn), os.path.join(out_dir, fn))
    print(f"{style}: train={n_train}, val={len(files)-n_train}")
