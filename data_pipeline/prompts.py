# -*- coding: utf-8 -*-
"""Prompt-template rendering and response parsing for the caption stage.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Reasoning models in the DeepSeek R1 family emit a scratchpad block that must
# be removed before the answer is parsed.
_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

REQUIRED_FIELDS = (
    "task",
    "source_caption",
    "instruction",
    "edited_caption",
    "edited_region",
    "edit_type",
)

VALID_EDIT_TYPES = ("word_swap", "refine", "region")


def strip_reasoning(text: str) -> str:
    """Remove R1 scratchpad blocks and stray fence markers."""
    if not text:
        return ""
    text = _THINK_BLOCK.sub(" ", text)
    return text.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """Return the first JSON object found in ``text``.

    Tries, in order: the whole string, the contents of a code fence, an object
    carved out by brace matching. Raises ``ValueError`` when nothing parses.
    """
    cleaned = strip_reasoning(text)
    if not cleaned:
        raise ValueError("model returned an empty reply")

    candidates: List[str] = [cleaned]
    candidates.extend(match.strip() for match in _FENCE.findall(cleaned))
    candidates.append(_brace_slice(cleaned))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"pairs": parsed}
    raise ValueError("no parsable JSON object in model reply")


def _brace_slice(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def render_axes(axes: Sequence[Any], rng: random.Random, take: int = 5) -> str:
    """Pick ``take`` axes and render them as bullet lines for the prompt."""
    if not axes:
        return "- (no constraint)"
    picks = list(axes)
    if len(picks) > take:
        picks = rng.sample(picks, take)
    lines = []
    for entry in picks:
        if isinstance(entry, dict):
            for key, value in entry.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {entry}")
    return "\n".join(lines)


def render_seeds(seeds: Iterable[Dict[str, Any]], rng: random.Random, take: int = 3) -> str:
    """Render seed captions as a numbered list."""
    pool = list(seeds)
    if not pool:
        return "1. (none)"
    if len(pool) > take:
        pool = rng.sample(pool, take)
    return "\n".join(f"{i + 1}. {seed['caption']}" for i, seed in enumerate(pool))


def seeds_for_task(seeds: Iterable[Dict[str, Any]], template_id: str) -> List[Dict[str, Any]]:
    """Filter the seed pool to those allowed under a task template."""
    allowed = [s for s in seeds if template_id in s.get("suitable_for", [])]
    return allowed or list(seeds)


def build_messages(
    prompts: Dict[str, Any],
    template_id: str,
    count: int,
    seed_text: str,
    axes_text: str,
    batch_index: int,
) -> List[Dict[str, str]]:
    """Assemble the chat messages for one caption-generation call."""
    templates = prompts.get("task_templates", {})
    if template_id not in templates:
        raise KeyError(f"no task template named '{template_id}' in caption_expansion.yaml")
    body = templates[template_id]["instruction"].format(
        count=count, axes=axes_text, seeds=seed_text, batch=batch_index
    )

    few_shot_note = ""
    examples = prompts.get("few_shot") or []
    if examples:
        rendered = "\n\n".join(
            json.dumps(example, ensure_ascii=False, indent=2) for example in examples
        )
        few_shot_note = (
            "\n\nTwo correct triples, for schema and wording only; do not copy their "
            "content:\n" + rendered
        )

    schema_note = "\n\nOutput schema:\n" + prompts.get("output_schema", "").strip()

    user_content = body + few_shot_note + schema_note
    return [
        {"role": "system", "content": prompts.get("system", "").strip()},
        {"role": "user", "content": user_content.strip()},
    ]


def normalise_pair(pair: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Coerce one returned triple into the canonical record shape."""
    if not isinstance(pair, dict):
        raise ValueError("pair is not an object")

    missing = [name for name in REQUIRED_FIELDS if not str(pair.get(name, "")).strip()]
    if missing:
        raise ValueError(f"pair is missing fields: {missing}")

    edit_type = str(pair["edit_type"]).strip().lower()
    if edit_type not in VALID_EDIT_TYPES:
        raise ValueError(f"unknown edit_type '{edit_type}'")

    return {
        "task": task_id,
        "subtask": str(pair.get("subtask", "")).strip() or "unspecified",
        "source_caption": _clean(pair["source_caption"]),
        "instruction": _clean(pair["instruction"]),
        "edited_caption": _clean(pair["edited_caption"]),
        "edited_region": _clean(pair["edited_region"]),
        "edit_type": edit_type,
    }


def _clean(text: Any) -> str:
    text = str(text).replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def parse_pairs(reply: str, task_id: str) -> List[Dict[str, Any]]:
    """Parse a model reply into validated triples, dropping malformed ones."""
    payload = extract_json_object(reply)
    raw_pairs = payload.get("pairs")
    if raw_pairs is None:
        # A model that returned a single object instead of the wrapper.
        raw_pairs = [payload] if "source_caption" in payload else []
    if not isinstance(raw_pairs, list):
        raise ValueError("'pairs' is not a list")

    accepted: List[Dict[str, Any]] = []
    for pair in raw_pairs:
        try:
            accepted.append(normalise_pair(pair, task_id))
        except ValueError:
            continue
    return accepted


def load_prompts(path: Optional[str] = None) -> Dict[str, Any]:
    from .config import load_caption_prompts

    return load_caption_prompts(path) if path else load_caption_prompts()
