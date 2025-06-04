# demo_compare_multi_style.py
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

# 关闭安全检测器
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# 2. 固定随机种子
generator = torch.Generator(device=device).manual_seed(42)

# 3. 要对比的风格和步数
styles = ["jojo", "cyberpunk2077", "ghibli"]
steps  = [25000, 30000, 35000, 40000]

# 4. 同一张内容图
init_image = (
    Image.open("demo_content.jpg")
         .convert("RGB")
         .resize((512,512))
)

# 5. 遍历生成并保存
for style in styles:
    for step in steps:
        # 5.1 加载对应 checkpoint 的全模型 UNet（FP16）
        ckpt_dir = f"outputs/checkpoint-{step}"
        new_unet = UNet2DConditionModel.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16
        ).to(device)
        pipe.unet = new_unet

        # 5.2 构造 prompt
        prompt = f"A portrait in the style of {style}"

        # 5.3 Img2Img 推理（固定 generator）
        out = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.7,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        # 5.4 保存结果
        out_file = f"demo_{style}_{step}.png"
        out.save(out_file)
        print(f"Saved {out_file}")
