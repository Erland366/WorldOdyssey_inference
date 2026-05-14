import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

# Available models: 
# 1. Wan-AI/Wan2.1-T2V-14B-Diffusers
# 2. Erland/tiny-wan2.1-t2v-debug
# 3. Wan-AI/Wan2.1-T2V-1.3B-Diffusers

model_id = "Erland/tiny-wan2.1-t2v-debug"
pipe = WanPipeline.from_pretrained(model_id, torch_dtype={
        "default" : torch.bfloat16,
        "vae" : torch.float32,
    },
    device_map="balanced"
)

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=15)
