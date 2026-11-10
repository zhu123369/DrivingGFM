# -*- coding: utf-8 -*-
"""Prompt-to-Prompt cross-attention control for Stable Diffusion 3.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # the controller itself is torch-only, its geometry helpers are not
    import torch

    _STORE_DTYPE: Any = torch.float16
except ImportError:  # allows the mask / alignment helpers to be used and tested
    torch = None  # type: ignore[assignment]
    _STORE_DTYPE = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "attention control needs PyTorch; install torch and diffusers to render pairs"
        )


@dataclass
class P2PConfig:
    """Knobs for one pair of runs."""

    edit_mode: str = "word_swap"          # word_swap | region | refine
    # Fraction of the denoising schedule, measured from the first step, over
    # which attention maps are replaced. Early steps fix the layout, late steps
    # add texture, so replacement is usually confined to the first half.
    replace_start: float = 0.0
    replace_end: float = 0.5
    # Blocks whose maps are recorded, counted from the input. Attention control
    # is most effective in the earlier blocks; storing every block is both slow
    # and unnecessary.
    store_blocks: Tuple[int, ...] = (0, 1, 2)
    # Weight of the source map in the blend. 1.0 means full replacement, which
    # is what Algorithm 2 describes; lower values keep some of the edited run's
    # own attention.
    self_replace_weight: float = 1.0
    store_dtype: Any = _STORE_DTYPE
    query_chunk: int = 1024
    # Spatial restriction for `region` mode: a boolean array over the latent
    # grid, True where replacement is allowed.
    region_mask: Optional[np.ndarray] = None


class AttentionController:
    """Holds the recorded maps and decides, step by step, what to do with them.

    One controller drives one pair of runs: first in ``store`` mode for the
    source prompt, then in ``replace`` mode for the edited prompt. ``step`` is
    advanced by the attention processors themselves, once per transformer
    forward, so it stays in sync with the pipeline without a callback.
    """

    def __init__(self, p2p: P2PConfig, num_inference_steps: int,
                 token_aligner: Optional["TokenAligner"] = None):
        self.cfg = p2p
        self.num_steps = max(1, int(num_inference_steps))
        self.mode = "off"
        self.step = -1
        self.calls = 0
        self.calls_per_step = 1
        self.maps: Dict[str, torch.Tensor] = {}
        self.token_aligner = token_aligner
        self.replace_pairs: List[Tuple[int, int]] = []
        self.stats = {"stored": 0, "replaced": 0, "bytes": 0}

    # -- lifecycle ---------------------------------------------------------
    def start_store(self, source_prompt: str, edited_prompt: str) -> None:
        """Begin recording the source run for the given prompt pair."""
        self.mode = "store"
        self.step = -1
        self.calls = 0
        self.maps.clear()
        self.replace_pairs = []
        if self.token_aligner is not None and self.cfg.edit_mode in ("word_swap", "region"):
            self.replace_pairs = self.token_aligner.mapping(source_prompt, edited_prompt)

    def start_replace(self) -> None:
        self.mode = "replace"
        self.step = -1
        self.calls = 0

    def finish(self) -> None:
        self.mode = "off"

    def tick(self) -> int:
        """Called once per joint-attention forward; returns the current step.

        Every transformer block is visited exactly once per denoising step, so
        dividing the call counter by the number of joint blocks keeps ``step``
        aligned with the pipeline without relying on a callback.
        """
        self.calls += 1
        self.step = (self.calls - 1) // max(1, self.calls_per_step)
        return self.step

    # -- policy ------------------------------------------------------------
    @property
    def replace_window(self) -> Tuple[int, int]:
        start = int(math.floor(self.cfg.replace_start * self.num_steps))
        end = int(math.ceil(self.cfg.replace_end * self.num_steps))
        return start, max(start, min(end, self.num_steps))

    def active_window(self) -> bool:
        start, end = self.replace_window
        return start <= self.step < end

    def should_store(self, block_index: int) -> bool:
        return (
            self.mode == "store"
            and block_index in self.cfg.store_blocks
            and self.active_window()
        )

    def should_replace(self, block_index: int) -> bool:
        return (
            self.mode == "replace"
            and block_index in self.cfg.store_blocks
            and self.active_window()
            and block_index in self.stored_blocks()
        )

    def stored_blocks(self) -> set:
        return {int(key.split("/")[0]) for key in self.maps}

    def map_key(self, block_index: int) -> str:
        return f"{block_index}/{self.step}"

    # -- storage -----------------------------------------------------------
    def store(self, block_index: int, text_slice: torch.Tensor) -> None:
        """Keep a text-conditioning slice, on CPU, outside the autograd graph."""
        _require_torch()
        key = self.map_key(block_index)
        payload = text_slice.detach().to("cpu", dtype=self.cfg.store_dtype)
        self.maps[key] = payload
        self.stats["stored"] += 1
        self.stats["bytes"] += payload.numel() * payload.element_size()

    def fetch(self, block_index: int) -> Optional[torch.Tensor]:
        return self.maps.get(self.map_key(block_index))

    def store_bytes_mb(self) -> float:
        return self.stats["bytes"] / (1024.0 * 1024.0)


class P2PJointAttnProcessor:
    """Joint-attention processor with explicit probabilities.

    Drop-in replacement for the SD3 joint attention processor: same call
    signature, same residual / projection / rescale handling, except that the
    text part of the attention matrix is materialised so it can be recorded and
    replaced.
    """

    def __init__(self, block_index: int, controller: AttentionController):
        self.block_index = block_index
        self.controller = controller
        self._warned_shape = False

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        self.controller.tick()

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and getattr(attn, "norm_cross", False):
            context = attn.norm_encoder_hidden_states(context)

        key = attn.to_k(context)
        value = attn.to_v(context)

        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]
        heads = attn.heads
        head_dim = query.shape[-1] // heads

        query = query.view(batch_size, q_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_len, heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_len, heads, head_dim).transpose(1, 2)

        # Text tokens precede latent tokens in the joint sequence; a plain
        # self-attention block (kv_len == q_len) has no text slice.
        n_text = kv_len - q_len if kv_len > q_len else 0

        if n_text <= 0:
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).reshape(batch_size, q_len, heads * head_dim)
        else:
            out = self._attend_with_control(query, key, value, n_text, attention_mask)

        out = out.to(query.dtype)
        if getattr(attn, "to_out", None) is not None:
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)

        if getattr(attn, "residual_connection", False) and residual is not None:
            out = out + residual
        return out / getattr(attn, "rescale_output_factor", 1.0)

    # -- explicit attention over the text slice ----------------------------
    def _attend_with_control(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        n_text: int,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, heads, q_len, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)
        chunk = max(1, int(self.controller.cfg.query_chunk))

        do_store = self.controller.should_store(self.block_index)
        do_replace = self.controller.should_replace(self.block_index)

        collected: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        for start in range(0, q_len, chunk):
            stop = min(start + chunk, q_len)
            logits = torch.matmul(query[:, :, start:stop], key.transpose(-2, -1)) * scale
            if attention_mask is not None:
                mask = self._broadcast_mask(attention_mask, batch_size, heads, stop - start, key.shape[2])
                logits = logits + mask
            probs = torch.softmax(logits.float(), dim=-1).to(query.dtype)

            text_slice = probs[:, :, :, :n_text]
            if do_store:
                collected.append(text_slice)

            if do_replace:
                text_slice = self._blend(start, stop, text_slice)
                probs = torch.cat([text_slice, probs[:, :, :, n_text:]], dim=-1)

            probs = probs.to(value.dtype)
            outputs.append(torch.matmul(probs, value))

        if do_store and collected:
            self.controller.store(self.block_index, torch.cat(collected, dim=2))

        out = torch.cat(outputs, dim=2)
        return out.transpose(1, 2).reshape(batch_size, q_len, heads * head_dim)

    def _blend(self, start: int, stop: int, text_slice: torch.Tensor) -> torch.Tensor:
        """Replace (or blend) the text columns of this query chunk."""
        stored = self.controller.fetch(self.block_index)
        if stored is None:
            return text_slice

        stored = stored.to(device=text_slice.device, dtype=text_slice.dtype)
        if stored.shape[2] < stop:
            self._warn_once(
                f"stored attention map covers {stored.shape[2]} queries but the "
                f"edit reaches query {stop}; skipping replacement for this block"
            )
            return text_slice

        source = stored[:, :, start:stop]
        if source.shape[-1] != text_slice.shape[-1]:
            self._warn_once(
                "stored and live text lengths differ, which happens when the "
                "source and edited prompts tokenise to different lengths; "
                "falling back to no replacement"
            )
            return text_slice

        weight = float(self.controller.cfg.self_replace_weight)
        pairs = self.controller.replace_pairs
        if pairs and self.controller.cfg.edit_mode in ("word_swap", "region") and weight < 1.0:
            # Keep the edited run's own attention everywhere, borrow the source
            # run's attention only for the columns the instruction touches.
            result = text_slice.clone()
            for edited_index, source_index in pairs:
                if edited_index < result.shape[-1] and source_index < source.shape[-1]:
                    blended = (
                        weight * source[:, :, :, source_index]
                        + (1.0 - weight) * result[:, :, :, edited_index]
                    )
                    result[:, :, :, edited_index] = blended
        else:
            result = weight * source + (1.0 - weight) * text_slice

        mask = self.controller.cfg.region_mask
        if mask is not None and self.controller.cfg.edit_mode == "region":
            result = self._apply_region_mask(result, mask, start, stop, text_slice)

        self.controller.stats["replaced"] += 1
        return result

    def _apply_region_mask(
        self,
        candidate: torch.Tensor,
        mask: np.ndarray,
        start: int,
        stop: int,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        flat = np.asarray(mask).reshape(-1)
        if flat.size < stop:
            self._warn_once(
                f"region mask covers {flat.size} latent positions but the "
                f"attention has {stop}; ignoring the mask"
            )
            return candidate
        inside = torch.from_numpy(flat[start:stop].astype(np.float32)).to(candidate.device)
        inside = inside.view(1, 1, -1, 1)
        return candidate * inside + fallback * (1.0 - inside)

    def _broadcast_mask(self, mask, batch_size, heads, q_chunk, kv_len):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[-1] != kv_len:
            mask = mask[..., :kv_len]
        return mask

    def _warn_once(self, message: str) -> None:
        if not self._warned_shape:
            self._warned_shape = True
            print(f"[p2p] warning: {message}")


class TokenAligner:
    """Maps edited-prompt token positions onto the source prompt's positions.

    Stable Diffusion 3 concatenates the text embeddings of its encoders along the
    sequence axis, so a token's column index in the joint attention is
    ``offset_of_encoder + local_index``. ``TokenAligner`` tokenises both prompts
    with each encoder's tokenizer, aligns the token strings with difflib, and
    returns ``(edited_index, source_index)`` pairs in that concatenated space.
    Those are the columns that get replaced in ``word_swap`` and ``region`` mode:
    the edited run borrows the source run's spatial attention for the concept
    that the instruction introduces.
    """

    def __init__(self, tokenizers: Sequence[Any], pad_lengths: Sequence[int]):
        if len(tokenizers) != len(pad_lengths):
            raise ValueError("tokenizers and pad_lengths must line up")
        self.tokenizers = list(tokenizers)
        self.pad_lengths = [int(n) for n in pad_lengths]
        self.offsets = [sum(self.pad_lengths[:i]) for i in range(len(self.pad_lengths))]

    @property
    def total_length(self) -> int:
        return sum(self.pad_lengths)

    def mapping(self, source_prompt: str, edited_prompt: str) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for tokenizer, offset, pad in zip(self.tokenizers, self.offsets, self.pad_lengths):
            pairs.extend(self._encoder_pairs(tokenizer, offset, pad, source_prompt, edited_prompt))
        return pairs

    def _encoder_pairs(
        self,
        tokenizer,
        offset: int,
        pad: int,
        source_prompt: str,
        edited_prompt: str,
    ) -> List[Tuple[int, int]]:
        source_tokens = self._tokens(tokenizer, source_prompt, pad)
        edited_tokens = self._tokens(tokenizer, edited_prompt, pad)
        matcher = difflib.SequenceMatcher(a=source_tokens, b=edited_tokens, autojunk=False)
        pairs: List[Tuple[int, int]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    pairs.append((offset + j1 + k, offset + i1 + k))
            elif tag == "insert":
                # A brand-new concept has no source twin; borrow the attention of
                # the token it was inserted after (the anchored element).
                anchor = max(0, i1 - 1)
                for j in range(j1, j2):
                    pairs.append((offset + j, offset + anchor))
            # `delete` needs no entry: the source column is simply unused.
        return [pair for pair in pairs if 0 <= pair[0] < pad and 0 <= pair[1] < pad]

    @staticmethod
    def _tokens(tokenizer, prompt: str, pad: int) -> List[str]:
        ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=pad,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        tokens: List[str] = []
        for token_id in ids:
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            tokens.append(text.strip())
        return tokens


def install_p2p(transformer, controller: AttentionController) -> int:
    """Swap the joint attention processors for P2P-aware ones.

    Only keys of the form ``...transformer_blocks.<i>...`` are wrapped: those are
    the blocks that attend over text and latents jointly. Single-stream blocks
    (``single_transformer_blocks``) and any other processor are left as they are,
    because they carry no text-conditioning map to exchange.

    Returns the number of processors replaced. The controller is told how many
    blocks report per forward so it can keep its step counter aligned.
    """
    _require_torch()
    processors = transformer.attn_processors
    mapping: Dict[str, Any] = {}
    installed = 0
    for name, processor in processors.items():
        block_index = _parse_block_index(name)
        if block_index is None:
            mapping[name] = processor
            continue
        mapping[name] = P2PJointAttnProcessor(block_index, controller)
        installed += 1

    if installed == 0:
        raise RuntimeError(
            "no joint attention blocks found; the transformer architecture is not "
            "the SD3 MMDiT this controller was written for"
        )

    controller.calls_per_step = installed
    transformer.set_attn_processor(mapping)
    return installed


def _parse_block_index(name: str) -> Optional[int]:
    """Return the index of a joint transformer block, or None if not one."""
    parts = name.split(".")
    for position, part in enumerate(parts[:-1]):
        if part == "transformer_blocks" and parts[position + 1].isdigit():
            return int(parts[position + 1])
    return None


def build_region_mask(
    box: Optional[Sequence[float]],
    grid_h: int,
    grid_w: int,
) -> Optional[np.ndarray]:
    """Turn a normalised ``(x1, y1, x2, y2)`` box into a latent-grid mask."""
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    col_start = max(0, int(round(min(x1, x2) * grid_w)))
    col_end = min(grid_w, max(col_start + 1, int(round(max(x1, x2) * grid_w))))
    row_start = max(0, int(round(min(y1, y2) * grid_h)))
    row_end = min(grid_h, max(row_start + 1, int(round(max(y1, y2) * grid_h))))
    mask[row_start:row_end, col_start:col_end] = True
    return mask
