# -*- coding: utf-8 -*-
"""Usage:
    python infer.py --image scene.png --prompt "make it a rainy night road scene"
    python infer.py --image scene.png --prompt "..." --device cuda --dtype float16 --steps 30 --seed 2026
"""

import argparse
import time

import torch
from PIL import Image

from pipeline_drivinggfm import DrivinggfmPipeline

MODEL_ID = "ZehongZhu/drivinggfm"


def parse_args():
    p = argparse.ArgumentParser(description="DrivingGFM inference")
    p.add_argument("--model", default=MODEL_ID, help="Hugging Face repo id or a local weights directory")
    p.add_argument("--image", required=True, help="path to the input scene image")
    p.add_argument("--prompt", required=True, help="editing instruction")
    p.add_argument("--steps", type=int, default=20, help="denoising steps")
    p.add_argument("--image-guidance", type=float, default=1.5, help="structural fidelity to the source image")
    p.add_argument("--guidance", type=float, default=7.5, help="instruction adherence")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--size", type=int, default=512, help="inference resolution (short edge scaled, centre-cropped)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16"])
    p.add_argument("--out", default="output.png")
    return p.parse_args()


def main():
    a = parse_args()

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (a.dtype == "auto" and device == "cuda") else (
        torch.float32 if a.dtype == "auto" else getattr(torch, a.dtype))

    print(f"[DrivingGFM] model={a.model} device={device} dtype={dtype}")
    t0 = time.time()
    pipe = DrivinggfmPipeline.from_pretrained(a.model, torch_dtype=dtype, trust_remote_code=True)
    pipe.to(device)
    if device == "cpu":
        pipe.enable_attention_slicing()
    print(f"[DrivingGFM] loaded in {time.time()-t0:.1f}s")

    img = Image.open(a.image).convert("RGB")
    w, h = img.size
    s = a.size / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    cw, ch = img.size
    left, top = (cw - a.size) // 2, (ch - a.size) // 2
    img = img.crop((left, top, left + a.size, top + a.size))

    t0 = time.time()
    out = pipe.drive(img, a.prompt, num_inference_steps=a.steps,
                     image_guidance_scale=a.image_guidance,
                     guidance_scale=a.guidance, seed=a.seed)
    print(f"[DrivingGFM] generated in {time.time()-t0:.1f}s")
    out.save(a.out)
    print(f"[DrivingGFM] saved to {a.out}")


if __name__ == "__main__":
    main()
