# DrivingGFM

**DrivingGFM** is a generalist foundation model for unified driving visual perception. Built upon Stable Diffusion V3 with instruction-tuning and DPO-based reinforcement learning, it supports multiple driving visual perception tasks including object detection, image enhancement, super-resolution and object removal via editing instructions.

Weights are distributed on Hugging Face: [`ZehongZhu/drivinggfm`](https://huggingface.co/ZehongZhu/drivinggfm)

## Repository layout

```
.
├── pipeline_drivinggfm.py  
├── infer.py                
├── requirements.txt
└── LICENSE                  
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
import torch
from PIL import Image
from pipeline_drivinggfm import DrivinggfmPipeline

pipe = DrivinggfmPipeline.from_pretrained(
    "ZehongZhu/drivinggfm", torch_dtype=torch.float16, trust_remote_code=True
)
pipe.to("cuda")

out = pipe.drive(
    Image.open("scene.png"),
    "make it a rainy night road scene",
    num_inference_steps=20,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
    seed=2026,
)
out.save("result.png")
```

A local weights directory works just as well — pass the directory instead of the repo id:

```python
pipe = DrivinggfmPipeline.from_pretrained("/path/to/drivinggfm", trust_remote_code=True)
```

## Command line

```bash
python infer.py --image scene.png --prompt "add a right-turn sign on the roadside"
python infer.py --image scene.png --prompt "..." --device cuda --dtype float16 --steps 30 --seed 2026
```

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | 20 | Denoising steps; 20–30 balances quality and speed |
| `--image-guidance` | 1.5 | Structural fidelity: higher stays closer to the source geometry, typically 1.0–2.0 |
| `--guidance` | 7.5 | Instruction adherence: higher follows the prompt more literally, but over-saturates |
| `--size` | 512 | Inference resolution (short edge scaled, centre-cropped) |
| `--seed` | 2026 | Random seed |

## Loading notes

`model_index.json` declares `_class_name` as `["pipeline_drivinggfm", "Drivinggfm"]`, which tells diffusers to take the pipeline implementation from `pipeline_drivinggfm.py` in the model repository instead of its own registry. Both entry points below were verified against the published weights:

```python
# 1. Through the generic loader — needs trust_remote_code for the repo-hosted pipeline code
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("ZehongZhu/drivinggfm", trust_remote_code=True)

# 2. Through the class shipped in this repository
from pipeline_drivinggfm import DrivinggfmPipeline
pipe = DrivinggfmPipeline.from_pretrained("ZehongZhu/drivinggfm", trust_remote_code=True)
```

Omitting `trust_remote_code=True` raises `The repository ... contains custom code which must be executed`.

`DrivinggfmPipeline` inherits only the generic `DiffusionPipeline` base class and implements the whole sampling stack itself — text/image prompt encoding, the dual-guidance denoising loop over an 8-channel UNet input, and the VAE decode — so it does not depend on any prebuilt diffusers pipeline class. Both `image_guidance_scale` (structural fidelity to the source image) and `guidance_scale` (instruction adherence) are therefore available as first-class parameters. Weight, VAE, CLIP text encoder, tokenizer, and the Euler Ancestral scheduler all load through the standard `from_pretrained` sub-directory conventions.

## Model composition

| Component | Implementation | Parameters |
|---|---|---|
| UNet | `UNet2DConditionModel`, in_channels=8, cross_attn_dim=768 | 859.5 M |
| VAE | `AutoencoderKL`, 4 latent channels, 512 px training resolution | 83.7 M |
| Text encoder | `CLIPTextModel` (ViT-L/14 text tower) | 123.1 M |
| Scheduler | `EulerAncestralDiscreteScheduler` | — |

## Modifications

The distributed weight and configuration files were modified from the reference checkpoint; this release is a derived one. The notice is recorded in the Hugging Face repo under `model_index.json` → `_drivinggfm.modified`, in accordance with Section 4(c) of the CreativeML Open RAIL-M license, which requires modified files to carry a prominent notice that they were changed.

## License

Weights and code are distributed under **CreativeML Open RAIL-M**; the full text is in [LICENSE](LICENSE). In short:

- Commercial use, modification, and redistribution are permitted;
- Any redistribution must carry the full license text, retain the copyright and attribution notices, and bind downstream users to the same Attachment A use restrictions;
- Modified files must be marked as modified. The modifications relative to the reference checkpoint are recorded in the Hugging Face repo under `model_index.json` → `_drivinggfm.modified`.

## Citation

```bibtex
@misc{drivinggfm2026,
  title  = {DrivingGFM: Generalist Foundation Model for Unified Driving Visual Perception},
  author = {DrivingGFM Team, Tongji University},
  year   = {2026},
  note   = {Model v1.0}
}
```
