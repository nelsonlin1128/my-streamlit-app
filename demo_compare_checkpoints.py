# demo_compare_checkpoints.py
import os
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from PIL import Image
import torch

device = "cuda"

# 1. 初始化 Img2Img 管线（FP16）
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

# 2. 循环各个 checkpoint
for step in [5000, 10000, 15000, 20000]:
    ckpt_dir = f"outputs/checkpoint-{step}"
    # 加载全模型 UNet 时指定 torch_dtype=torch.float16
    new_unet = UNet2DConditionModel.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.float16
    ).to(device)
    # 或者用 new_unet.half() 也可以：
    # new_unet = new_unet.half().to(device)

    pipe.unet = new_unet  # 替换管线里的 UNet

    # 准备内容图
    init_image = (
        Image.open("demo_content.jpg")
             .convert("RGB")
             .resize((512,512))
    )

    # 执行 img2img 推理
    out = pipe(
        prompt="A portrait in the style of jojo",
        image=init_image,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]

    out.save(f"demo_jojo_{step}.png")
    print(f"Saved demo_jojo_{step}.png")
