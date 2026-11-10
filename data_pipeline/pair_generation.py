# -*- coding: utf-8 -*-
"""Stage 3: render paired source/edited images with Stable Diffusion 3 + P2P.

Two strategies are supported:

``p2p``
    Both frames are denoised from one shared noise sample. Used for nine of the
    ten tasks.

``degradation``
    One image is generated and the source frame is produced by degrading it (a
    4x bicubic down/up-sampling for super-resolution). Kept as an offline
    fallback: it gives exact supervision for super-resolution without depending
    on the attention controller, which is useful on machines where the extra
    memory of a stored attention pass is not affordable.

Nothing is generated unless the triple already passed the text filters, so a
broken caption never costs a GPU run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg
from .p2p import AttentionController, P2PConfig, TokenAligner, build_region_mask, install_p2p


DEFAULT_MODEL = "stabilityai/stable-diffusion-3-medium"


@dataclass
class SamplingSettings:
    num_inference_steps: int = 25
    guidance_scale: float = 4.5
    height: int = 512
    width: int = 512
    max_sequence_length: int = 77
    negative_prompt: str = "blurry, low quality, distorted, watermark, text overlay"
    replace_start: float = 0.0
    replace_end: float = 0.5
    store_blocks: Tuple[int, ...] = (0, 1, 2)
    self_replace_weight: float = 1.0
    dtype: str = "float16"
    device: str = "cuda"


class PairGenerator:
    """Wraps a loaded SD3 pipeline and produces paired images."""

    def __init__(self, model_id: str = DEFAULT_MODEL, settings: Optional[SamplingSettings] = None):
        self.settings = settings or SamplingSettings()
        self.pipe = self._load(model_id)
        self.transformer = self.pipe.transformer
        self.aligner = self._build_aligner()

    # -- setup -------------------------------------------------------------
    def _load(self, model_id: str):
        import torch
        from diffusers import StableDiffusion3Pipeline

        dtype = getattr(torch, self.settings.dtype)
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.to(self.settings.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _build_aligner(self) -> TokenAligner:
        tokenizers = [getattr(self.pipe, name) for name in ("tokenizer", "tokenizer_2", "tokenizer_3")
                      if getattr(self.pipe, name, None) is not None]
        pad = [self.settings.max_sequence_length] * len(tokenizers)
        return TokenAligner(tokenizers, pad)

    def latent_grid(self) -> Tuple[int, int]:
        """Latent resolution for the configured image size."""
        scale = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        return self.settings.height // scale, self.settings.width // scale

    # -- generation --------------------------------------------------------
    def generate_pair(
        self,
        record: Dict[str, Any],
        edit_mode: str,
        pair_strategy: str = "p2p",
        region_box: Optional[List[float]] = None,
        seed: int = 2026,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Return ``(source_image, edited_image, run_metadata)``."""
        import torch

        s = self.settings
        task = record["task"]
        source_caption = record["source_caption"]
        edited_caption = record["edited_caption"]

        if pair_strategy == "degradation":
            edited_image = self._sample(edited_caption, seed)
            source_image = degrade_image(edited_image, factor=4)
            meta = {
                "strategy": "degradation",
                "seed": seed,
                "steps": s.num_inference_steps,
                "guidance_scale": s.guidance_scale,
                "degradation": "bicubic_downsample_4x",
            }
            return source_image, edited_image, meta

        mask = None
        grid_h, grid_w = self.latent_grid()
        if edit_mode == "region" and region_box is not None:
            mask = build_region_mask(region_box, grid_h, grid_w)

        p2p_cfg = P2PConfig(
            edit_mode=edit_mode,
            replace_start=s.replace_start,
            replace_end=s.replace_end,
            store_blocks=s.store_blocks,
            self_replace_weight=s.self_replace_weight,
            region_mask=mask,
        )
        controller = AttentionController(p2p_cfg, s.num_inference_steps, token_aligner=self.aligner)
        installed = install_p2p(self.transformer, controller)

        try:
            controller.start_store(source_caption, edited_caption)
            source_image = self._sample(source_caption, seed)
            store_mb = controller.store_bytes_mb()

            controller.start_replace()
            edited_image = self._sample(edited_caption, seed)
        finally:
            controller.finish()
            controller.maps.clear()
            # Put the original processors back, so a failure cannot leave the
            # shared pipeline instrumented for the next pair.
            self._restore_processors()

        meta = {
            "strategy": "p2p",
            "edit_mode": edit_mode,
            "seed": seed,
            "steps": s.num_inference_steps,
            "guidance_scale": s.guidance_scale,
            "replace_start": s.replace_start,
            "replace_end": s.replace_end,
            "replace_window": list(controller.replace_window),
            "store_blocks": list(s.store_blocks),
            "attention_blocks": installed,
            "stored_mb": round(store_mb, 1),
            "replace_pairs": len(controller.replace_pairs),
            "region_mask": bool(mask is not None),
            "task": task,
        }
        return source_image, edited_image, meta

    def _sample(self, prompt: str, seed: int):
        import torch

        s = self.settings
        generator = torch.Generator(device=s.device).manual_seed(int(seed))
        return self.pipe(
            prompt=prompt,
            negative_prompt=s.negative_prompt,
            num_inference_steps=s.num_inference_steps,
            guidance_scale=s.guidance_scale,
            height=s.height,
            width=s.width,
            max_sequence_length=s.max_sequence_length,
            generator=generator,
            output_type="pil",
        ).images[0]

    def _restore_processors(self) -> None:
        """Reload the stock attention processors after a pair."""
        try:
            self.transformer.set_default_attn_processor()
        except AttributeError:  # pragma: no cover - older diffusers
            self.transformer.set_attn_processor(
                {name: proc for name, proc in self.transformer.attn_processors.items()}
            )


def degrade_image(image, factor: int = 4):
    """Down- then up-sample to make a matched low-resolution source frame."""
    width, height = image.size
    small = image.resize(
        (max(1, width // factor), max(1, height // factor)), resample=_bicubic()
    )
    return small.resize((width, height), resample=_bicubic())


def _bicubic():
    from PIL import Image

    return Image.BICUBIC


# --------------------------------------------------------------------- driver
def pair_id_for(record: Dict[str, Any]) -> str:
    key = record.get("pair_key", "")
    if not key:
        raw = f"{record['task']}|{record['source_caption']}|{record['instruction']}"
        key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return key


def generate_task_pairs(
    task_id: str,
    records: List[Dict[str, Any]],
    generator: PairGenerator,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> int:
    """Render pairs for one task; returns how many were written."""
    out_dir = Path(out_dir) if out_dir else (cfg.pairs_dir() / task_id)
    cfg.ensure_dir(out_dir)
    ledger = cfg.pairs_dir() / f"{task_id}.jsonl"

    done = {record.get("pair_id") for record in cfg.read_jsonl(ledger)}
    written = 0
    failures = 0

    for index, record in enumerate(records):
        pid = pair_id_for(record)
        if pid in done:
            continue

        try:
            source_img, edited_img, meta = generator.generate_pair(
                record,
                edit_mode=record.get("edit_mode") or "word_swap",
                pair_strategy=record.get("pair_strategy") or "p2p",
                region_box=record.get("region_box"),
                seed=int(record.get("seed", 2026)),
            )
        except Exception as exc:  # a single bad pair must not kill a 5-hour run
            failures += 1
            if verbose:
                print(f"[{task_id}] pair {pid} failed: {type(exc).__name__}: {exc}")
            continue

        src_path = out_dir / f"{pid}_src.png"
        edit_path = out_dir / f"{pid}_edit.png"
        source_img.save(src_path)
        edited_img.save(edit_path)

        payload = {
            "pair_id": pid,
            "task": task_id,
            "source_caption": record["source_caption"],
            "instruction": record["instruction"],
            "edited_caption": record["edited_caption"],
            "edited_region": record.get("edited_region", ""),
            "subtask": record.get("subtask", ""),
            "image_source": str(src_path.relative_to(cfg.REPO_ROOT)),
            "image_edited": str(edit_path.relative_to(cfg.REPO_ROOT)),
            "gen": meta,
            "filters": {},
            "status": "generated",
        }
        cfg.append_jsonl(ledger, payload)
        written += 1

        if verbose and written % 50 == 0:
            print(f"[{task_id}] {written} pairs rendered ({failures} failures)")

    if verbose:
        print(f"[{task_id}] {written} pairs rendered, {failures} failures")
    return written
