# -*- coding: utf-8 -*-
"""Stage 1: expand seed captions into (caption, instruction, caption) triples.

This is the caption half of Section III-B of the manuscript. Seed captions are
handed to a reasoning model together with a task template; the model returns
triples that are validated against the schema and appended to a JSONL file,
one record per line, so the stage is resumable and auditable line by line.

"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import config as cfg
from .prompts import (
    build_messages,
    parse_pairs,
    render_axes,
    render_seeds,
    seeds_for_task,
)


# --------------------------------------------------------------------- client
@dataclass
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout: int = 180
    retries: int = 3
    backoff: float = 5.0

    @classmethod
    def from_env(cls, prompt_cfg: Dict[str, Any]) -> "LLMSettings":
        gen = prompt_cfg.get("generation_settings", {})
        base_url = os.environ.get("DRIVINGGFM_LLM_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("DRIVINGGFM_LLM_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError(
                "the caption stage needs DRIVINGGFM_LLM_BASE_URL and "
                "DRIVINGGFM_LLM_API_KEY; any OpenAI-compatible endpoint works"
            )
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=os.environ.get("DRIVINGGFM_LLM_MODEL", prompt_cfg.get("model", "deepseek-r1")),
            temperature=float(gen.get("temperature", 0.9)),
            top_p=float(gen.get("top_p", 0.95)),
            max_tokens=int(gen.get("max_tokens", 4096)),
            retries=int(gen.get("retries", 3)),
            backoff=float(gen.get("retry_backoff_seconds", 5.0)),
        )


class ChatClient:
    """Minimal chat-completions client built on urllib.

    Kept dependency-free on purpose: reproducing the caption stage should not
    require a vendor SDK.
    """

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0

    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        payload = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.retries + 1):
            request = urllib.request.Request(
                f"{self.settings.base_url}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                with self._lock:
                    self.calls += 1
                return _message_text(parsed)
            except urllib.error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", "ignore")[:400]
                # A 4xx other than rate limiting will not improve on retry.
                if 400 <= exc.code < 500 and exc.code != 429:
                    raise RuntimeError(f"LLM request rejected ({exc.code}): {detail}") from exc
                time.sleep(self.settings.backoff * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(self.settings.backoff * (attempt + 1))

        with self._lock:
            self.failures += 1
        raise RuntimeError(f"LLM request failed after retries: {last_error}")


def _message_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"response has no choices: {str(response)[:200]}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):  # some providers return content parts
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not content:
        raise RuntimeError("response content is empty")
    return content


# ------------------------------------------------------------------ dedup key
def pair_key(record: Dict[str, Any]) -> str:
    """Stable identity of a triple, used to avoid regenerating the same pair."""
    raw = "|".join(
        [
            record["task"],
            _norm(record["source_caption"]),
            _norm(record["instruction"]),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


# ------------------------------------------------------------------ generation
def _fetch_pairs(
    client: ChatClient,
    prompts: Dict[str, Any],
    template_id: str,
    task_id: str,
    index: int,
    seed: int,
    seed_pool: Sequence[Dict[str, Any]],
    axes_pool: Sequence[Any],
    ask: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed + index)
    seed_text = render_seeds(seed_pool, rng, take=3)
    axes_text = render_axes(axes_pool, rng, take=5)
    messages = build_messages(prompts, template_id, ask, seed_text, axes_text, index)
    return parse_pairs(client.complete(messages), task_id)


def _append_unique(
    path: Path,
    records: List[Dict[str, Any]],
    seen: set,
    lock: threading.Lock,
    gen_meta: Dict[str, Any],
) -> int:
    """Append new, non-duplicate records; returns how many were written."""
    written = 0
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                key = pair_key(record)
                if key in seen:
                    continue
                seen.add(key)
                payload = dict(record)
                payload["pair_key"] = key
                payload["gen"] = gen_meta
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
    return written


def expand_task(
    task_id: str,
    target: Optional[int] = None,
    *,
    workers: int = 8,
    seed: int = 2026,
    raw_path: Optional[Path] = None,
    prompts: Optional[Dict[str, Any]] = None,
    seeds: Optional[Sequence[Dict[str, Any]]] = None,
    tasks_cfg: Optional[cfg.PipelineConfig] = None,
    client: Optional[ChatClient] = None,
    verbose: bool = True,
) -> int:
    """Generate raw triples for one task until ``target`` distinct pairs exist.

    Returns the number of newly written records. Existing records in
    ``raw_path`` are counted first, so the call can be repeated safely.
    """
    tasks_cfg = tasks_cfg or cfg.load_tasks()
    spec = tasks_cfg.task(task_id)
    prompts = prompts or cfg.load_caption_prompts()
    seeds = list(seeds or cfg.load_seeds())

    target = int(target) if target is not None else spec.quota
    template_id = spec.prompt_template
    if template_id not in prompts.get("task_templates", {}):
        raise KeyError(f"task template '{template_id}' is not defined")

    raw_path = Path(raw_path) if raw_path else (cfg.captions_raw_dir() / f"{task_id}.jsonl")
    cfg.ensure_dir(raw_path.parent)

    existing = cfg.read_jsonl(raw_path)
    seen = {record.get("pair_key") or pair_key(record) for record in existing}
    written_before = len(existing)

    gen_settings = prompts.get("generation_settings", {})
    pairs_per_call = int(gen_settings.get("pairs_per_call", 8))
    # Ask for a couple of extra triples: the parser drops malformed ones, so a
    # batch asking for exactly n rarely yields n.
    ask = pairs_per_call + 2

    axes_pool = list(prompts.get("variety_axes", {}).get("common", [])) + list(
        prompts.get("variety_axes", {}).get(template_id, [])
    )
    seed_pool = seeds_for_task(seeds, template_id)

    if client is None:
        client = ChatClient(LLMSettings.from_env(prompts))

    gen_meta = {
        "stage": "caption_expansion",
        "model": client.settings.model,
        "temperature": client.settings.temperature,
        "top_p": client.settings.top_p,
        "template": template_id,
        "prompt_file": "prompts/caption_expansion.yaml",
        "seed_captions": "prompts/seeds.yaml",
        "seed": seed,
    }

    if written_before >= target:
        if verbose:
            print(f"[{task_id}] already has {written_before}/{target} raw triples, nothing to do")
        return 0

    state = {
        "written": written_before,
        "next_batch": 0,
        "consecutive_empty": 0,
        "failed_calls": 0,
        "written_new": 0,
    }
    lock = threading.Lock()
    stop = threading.Event()
    max_empty = int(
        cfg.load_filter_rules().get("quota_policy", {}).get(
            "max_consecutive_rejections_per_task", 12
        )
    )

    def worker() -> None:
        while not stop.is_set():
            with lock:
                if state["written"] >= target:
                    stop.set()
                    return
                index = state["next_batch"]
                state["next_batch"] += 1

            try:
                pairs = _fetch_pairs(
                    client, prompts, template_id, task_id,
                    index, seed, seed_pool, axes_pool, ask,
                )
            except RuntimeError:
                with lock:
                    state["failed_calls"] += 1
                    state["consecutive_empty"] += 1
                    if state["consecutive_empty"] >= max_empty:
                        stop.set()
                continue

            count = _append_unique(raw_path, pairs, seen, lock, gen_meta)
            with lock:
                state["written"] += count
                state["written_new"] += count
                state["consecutive_empty"] = 0 if pairs else state["consecutive_empty"] + 1
                if state["consecutive_empty"] >= max_empty:
                    stop.set()

            if verbose and state["written_new"] % 200 < count:
                print(f"[{task_id}] {state['written']}/{target} raw triples")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in range(max(1, workers)):
            pool.submit(worker)

    if verbose:
        print(
            f"[{task_id}] raw stage finished: {state['written']}/{target} "
            f"(+{state['written_new']}), {client.calls} LLM calls, "
            f"{client.failures} failures"
        )
    return state["written_new"]
