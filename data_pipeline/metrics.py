# -*- coding: utf-8 -*-
"""Metrics used by the filters and by the dataset quality report.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------- helpers
def to_float_array(image: Any) -> np.ndarray:
    """Accept PIL images, uint8 arrays or float arrays; return HxWx3 in [0, 1]."""
    if hasattr(image, "convert"):  # PIL.Image
        image = np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    array = array.astype(np.float32)
    if array.max(initial=0.0) > 1.0 + 1e-6:
        array = array / 255.0
    return array


def luma(image: Any) -> np.ndarray:
    array = to_float_array(image)
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    points = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel = np.exp(-(points ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _filter_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
    # Move the window axis last, then contract with the kernel.
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def _gaussian_blur(array: np.ndarray, size: int = 11, sigma: float = 1.5) -> np.ndarray:
    kernel = _gaussian_kernel(size, sigma)
    array = _filter_axis(array, kernel, 0)
    return _filter_axis(array, kernel, 1)


def ssim(source: Any, edited: Any, data_range: float = 1.0) -> float:
    """Mean SSIM over the frame, computed on luma."""
    x = luma(source)
    y = luma(edited)
    if x.shape != y.shape:
        raise ValueError(f"ssim needs equal sizes, got {x.shape} and {y.shape}")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = _gaussian_blur(x)
    mu_y = _gaussian_blur(y)
    mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x_sq = _gaussian_blur(x * x) - mu_x_sq
    sigma_y_sq = _gaussian_blur(y * y) - mu_y_sq
    sigma_xy = _gaussian_blur(x * y) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(numerator / denominator))


def clip_directional_similarity(
    source_embedding: np.ndarray,
    edited_embedding: np.ndarray,
    text_embedding: np.ndarray,
) -> float:
    """Cos(direction of change, instruction), all vectors L2-normalised."""
    source = _l2(source_embedding)
    edited = _l2(edited_embedding)
    text = _l2(text_embedding)
    direction = edited - source
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(direction / norm, text))


def _l2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else vector


# ------------------------------------------------------------- photometric rules
def photometric_stats(image: Any) -> Dict[str, float]:
    array = to_float_array(image)
    grey = luma(array)
    channel_means = [float(array[..., c].mean()) for c in range(3)]
    return {
        "mean_luma": float(grey.mean()),
        "std_luma": float(grey.std()),
        "channel_mean_r": channel_means[0],
        "channel_mean_g": channel_means[1],
        "channel_mean_b": channel_means[2],
    }


def channel_shift(source: Any, edited: Any) -> float:
    """Largest absolute mean-channel difference between the two frames."""
    a = to_float_array(source)
    b = to_float_array(edited)
    if a.shape != b.shape:
        return float("inf")
    diffs = [abs(float(a[..., c].mean() - b[..., c].mean())) for c in range(3)]
    return max(diffs)


# --------------------------------------------------------------- edit localisation
def _erode(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    radius = kernel // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    return windows.all(axis=(-1, -2))


def _dilate(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    radius = kernel // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    return windows.any(axis=(-1, -2))


def _largest_component_share(mask: np.ndarray, max_side: int = 128) -> float:
    """Share of True pixels inside the largest 4-connected component.

    The mask is decimated to at most ``max_side`` on its long edge first so that
    a 512x512 frame costs a few milliseconds; the share is a coarse quantity by
    design, it only has to separate "one blob" from "speckle everywhere".
    """
    if not mask.any():
        return 0.0
    stride = max(1, math.ceil(max(mask.shape) / max_side))
    small = mask[::stride, ::stride]
    if not small.any():
        return 0.0

    visited = np.zeros_like(small, dtype=bool)
    best = 0
    height, width = small.shape
    for row in range(height):
        for col in range(width):
            if not small[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and small[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            best = max(best, size)
    return best / float(small.sum())


def edit_region_stats(
    source: Any,
    edited: Any,
    pixel_threshold: float = 0.06,
    opening_kernel: int = 3,
) -> Dict[str, float]:
    """Describe the changed region: how much, and how concentrated."""
    a = to_float_array(source)
    b = to_float_array(edited)
    if a.shape != b.shape:
        return {"area_fraction": 0.0, "largest_component_share": 0.0, "bbox": None}

    difference = np.abs(a - b).mean(axis=-1)
    raw = difference > pixel_threshold
    mask = _dilate(_erode(raw, opening_kernel), opening_kernel) if opening_kernel > 1 else raw

    area = float(mask.mean())
    share = _largest_component_share(mask)

    bbox = None
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        height, width = mask.shape
        bbox = [
            float(cols[0]) / width,
            float(rows[0]) / height,
            float(cols[-1] + 1) / width,
            float(rows[-1] + 1) / height,
        ]

    return {
        "area_fraction": area,
        "largest_component_share": share,
        "bbox": bbox,
        "mean_abs_diff": float(difference.mean()),
    }


# -------------------------------------------------------------- image duplicates
def dhash(image: Any, hash_size: int = 8) -> np.ndarray:
    """Difference hash: a bit per neighbour comparison on a (size+1)x size grid."""
    grey = luma(image)
    height, width = grey.shape
    rows = np.linspace(0, height - 1, hash_size + 1).round().astype(int)
    cols = np.linspace(0, width - 1, hash_size).round().astype(int)
    sample = grey[np.ix_(rows, cols)]
    return (sample[1:, :] > sample[:-1, :]).reshape(-1)


def hamming_distance(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape:
        return 10 ** 6
    return int(np.count_nonzero(first != second))


# ---------------------------------------------------------------- text metrics
def token_f1(first: str, second: str) -> float:
    """Lexical fallback for semantic F1 when BERTScore is unavailable."""
    a = set(_tokens(first))
    b = set(_tokens(second))
    if not a or not b:
        return 0.0
    precision = len(a & b) / len(a)
    recall = len(a & b) / len(b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tokens(text: str) -> List[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


class BertScorer:
    """BERTScore-F1 with an explicit, flagged fallback.

    ``available`` is False when the ``bert_score`` package or its weights are
    missing; callers must then record ``metric: token_f1`` in the sample metadata
    so the report never mixes the two.
    """

    def __init__(self, model: str = "distilbert-base-uncased", enabled: bool = True):
        self.model = model
        self.scorer = None
        self.available = False
        if enabled:
            try:
                from bert_score import BERTScorer

                self.scorer = BERTScorer(model_type=model, lang="en", rescale_with_baseline=False)
                self.available = True
            except Exception:  # missing package, missing weights, or offline run
                self.scorer = None
                self.available = False

    def score(self, first: str, second: str) -> Tuple[float, str]:
        if self.scorer is None:
            return token_f1(first, second), "token_f1"
        _, _, f1 = self.scorer.score([first], [second])
        return float(f1[0]), "bertscore_f1"


class ClipScorer:
    """CLIP image/text embeddings for CLIP-D."""

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda", enabled: bool = True):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.available = False
        if enabled:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                self.torch = torch
                self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
                self.processor = CLIPProcessor.from_pretrained(model_id)
                self.available = True
            except Exception:
                self.model = None
                self.processor = None
                self.available = False

    def embed(self, images: Sequence[Any], texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.available:
            raise RuntimeError("CLIP is not available; install transformers and torch")
        import torch

        image_inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        text_inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
        return image_features.cpu().numpy(), text_features.cpu().numpy()

    def directional_scores(
        self,
        sources: Sequence[Any],
        edits: Sequence[Any],
        instructions: Sequence[str],
    ) -> List[float]:
        """CLIP-D for a batch, computed on unit-normalised embeddings."""
        if len(sources) != len(edits):
            raise ValueError("sources and edits must have the same length")
        source_emb, _ = self.embed(sources, [""] * len(sources))
        edit_emb, text_emb = self.embed(edits, instructions)
        return [
            clip_directional_similarity(source_emb[i], edit_emb[i], text_emb[i])
            for i in range(len(sources))
        ]


# ------------------------------------------------------------------- dataset FID
def fid_to_reference(
    image_dir: str,
    reference: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> Optional[float]:
    """FID against a reference image folder, when ``torch-fidelity`` is installed.

    Returns None rather than raising, because FID is a reporting extra: a
    machine without the optional dependency should still be able to build and
    filter the dataset.
    """
    try:
        import torch_fidelity
    except ImportError:
        return None
    metrics = torch_fidelity.calculate_metrics(
        input1=image_dir,
        input2=reference,
        cuda=device.startswith("cuda"),
        fid=True,
        batch_size=batch_size,
        verbose=False,
    )
    return float(metrics.get("frechet_inception_distance", float("nan")))
