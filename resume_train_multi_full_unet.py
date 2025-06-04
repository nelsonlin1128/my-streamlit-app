# resume_train_multi_full_unet.py
import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 1. 加速器与设备
accelerator = Accelerator(mixed_precision="fp16")
device      = accelerator.device
print("CUDA available:", torch.cuda.is_available())
print("Accelerator device:", device)
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_properties(device).name)

# 2. 重新加载 20k 步的 UNet
unet = UNet2DConditionModel.from_pretrained(
    "outputs/checkpoint-20000",
    
).to(device)

# 3. 保持其他组件不变
vae       = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae"
).to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
)

# 冻结 VAE 与文本编码器
for p in vae.parameters():    p.requires_grad = False
for p in text_enc.parameters(): p.requires_grad = False

# 4. 数据定义（与原脚本一致）
torch.manual_seed(42)
class MultiStyleDataset(Dataset):
    def __init__(self, root_dir, tokenizer, max_length=77):
        self.samples = []
        for style in os.listdir(root_dir):
            style_dir = os.path.join(root_dir, style)
            for fn in os.listdir(style_dir):
                if fn.lower().endswith(".jpg"):
                    self.samples.append((os.path.join(style_dir, fn), style))
        self.tokenizer  = tokenizer
        self.tf         = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, style = self.samples[idx]
        img   = Image.open(path).convert("RGB")
        pixel = self.tf(img)
        style_name = style.replace("_512", "")
        prompt     = f"a portrait in the style of {style_name}"
        inputs = self.tokenizer(
            prompt,
            padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "pixel_values":   pixel,
            "input_ids":      inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze()
        }

# 5. 构建 DataLoader
train_root   = "data/multi_style/train"
train_ds     = MultiStyleDataset(train_root, tokenizer)
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda b: {
        "pixel_values":   torch.stack([x["pixel_values"]   for x in b]),
        "input_ids":      torch.stack([x["input_ids"]      for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b])
    }
)

# 6. 优化器 & 准备
optimizer = AdamW(unet.parameters(), lr=5e-6, weight_decay=0.01)
unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

# 7. TensorBoard（继续用原来的 log 目录）
writer = SummaryWriter(log_dir="runs/multi_style_fullunet")

# 8. 续训循环参数
start_step          = 20000
max_train_steps     = 40000
checkpoint_interval = 1000
global_step         = start_step

print(f"Resuming training from step {start_step} to {max_train_steps}…")
unet.train()

while global_step < max_train_steps:
    for batch in train_loader:
        optimizer.zero_grad()
        pixels    = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)

        # encode → add noise → UNet predict
        latents = vae.encode(pixels).latent_dist.sample() * 0.18215
        noise   = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        )
        noisy = scheduler.add_noise(latents, noise, timesteps)

        enc       = text_enc(input_ids, attention_mask=mask)[0]
        noise_pred= unet(noisy, timesteps, encoder_hidden_states=enc).sample

        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        optimizer.step()

        global_step += 1
        if global_step % 100 == 0:
            print(f"Step {global_step}/{max_train_steps}, Loss: {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), global_step)

        if global_step % checkpoint_interval == 0:
            ckpt_dir = f"outputs/checkpoint-{global_step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            unet.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

        if global_step >= max_train_steps:
            break

print("Resume training done. Saving final model…")
unet.save_pretrained("outputs/multi_style_unet_full")
writer.close()
print("✅ Model and logs saved.")
