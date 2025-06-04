# app.py
import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel

# —— Streamlit 頁面設定 ——
st.set_page_config(page_title="多風格 Img2Img Demo", layout="wide")
st.title("🎨 多風格藝術風格轉換")

# —— 側邊欄：參數選擇 ——
st.sidebar.header("參數設定")
style = st.sidebar.selectbox("選擇風格", ["jojo", "cyberpunk2077", "ghibli"] )
strength = st.sidebar.slider("風格強度 (strength)", 0.0, 1.0, 0.7)
guidance = st.sidebar.slider("引導強度 (guidance scale)", 1.0, 12.0, 7.5)
steps = st.sidebar.slider("推理步數 (inference steps)", 10, 100, 50)

# —— 上傳內容圖 ——
uploaded = st.file_uploader("上傳內容圖 (512×512)", type=["jpg","png"])
if not uploaded:
    st.warning("請先上傳一張圖片！")
    st.stop()
init_image = Image.open(uploaded).convert("RGB").resize((512,512))
st.image(init_image, caption="原圖", use_column_width=True)

# —— 快取管線載入 ——
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

pipe = load_pipeline()
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# —— 風格與 checkpoint 步數對應 ——
ckpt_map = {"jojo": 35000, "cyberpunk2077": 35000, "ghibli": 35000}
ckpt_step = ckpt_map.get(style, 15000)
ckpt_dir = f"outputs/checkpoint-{ckpt_step}"

# —— 快取 UNet 載入 ——
@st.cache_resource
def load_unet(ckpt_dir):
    new_unet = UNet2DConditionModel.from_pretrained(
        ckpt_dir, torch_dtype=torch.float16
    ).to("cuda")
    return new_unet

pipe.unet = load_unet(ckpt_dir)

# —— 生成按鈕與推理 ——
if st.sidebar.button("開始生成"):
    with st.spinner("生成中，請稍候…"):
        generator = torch.Generator(device="cuda").manual_seed(42)
        prompt = f"a portrait in the style of {style}"
        out = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator
        ).images[0]
        st.image(out, caption=f"{style} 風格化結果 (checkpoint {ckpt_step})", use_column_width=True)
