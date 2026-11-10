# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
PROMPT_DIR = REPO_ROOT / "prompts"

TASKS_CONFIG = CONFIG_DIR / "tasks.yaml"
FILTER_CONFIG = CONFIG_DIR / "filter_rules.yaml"
CAPTION_PROMPTS = PROMPT_DIR / "caption_expansion.yaml"
SEEDS_FILE = PROMPT_DIR / "seeds.yaml"
VQA_PROMPTS = PROMPT_DIR / "vqa_preference.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "PyYAML is required to read the pipeline configuration; "
            "install it with `pip install pyyaml`."
        ) from exc
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


@dataclass
class TaskSpec:
    """One of the ten perception tasks and its share of the training set."""

    id: str
    family: str
    quota: int
    edit_mode: str
    pair_strategy: str
    prompt_template: str
    filter_profile: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "family": self.family,
            "quota": self.quota,
            "edit_mode": self.edit_mode,
            "pair_strategy": self.pair_strategy,
            "prompt_template": self.prompt_template,
            "filter_profile": self.filter_profile,
        }
        payload.update(self.extra)
        return payload


class PipelineConfig:
    """Parsed view over ``configs/tasks.yaml``."""

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self.meta: Dict[str, Any] = raw.get("meta", {})
        self.tasks: List[TaskSpec] = []
        for entry in raw.get("tasks", []):
            known = {
                "id", "family", "quota", "edit_mode", "pair_strategy",
                "prompt_template", "filter_profile",
            }
            missing = {"id", "quota", "edit_mode", "pair_strategy"} - set(entry)
            if missing:
                raise ValueError(
                    f"task entry {entry.get('id', '<unnamed>')} is missing {sorted(missing)}"
                )
            extra = {k: v for k, v in entry.items() if k not in known}
            self.tasks.append(
                TaskSpec(
                    id=entry["id"],
                    family=entry.get("family", "unknown"),
                    quota=int(entry["quota"]),
                    edit_mode=entry["edit_mode"],
                    pair_strategy=entry["pair_strategy"],
                    prompt_template=entry.get("prompt_template", entry["id"]),
                    filter_profile=entry.get("filter_profile", "generic"),
                    extra=extra,
                )
            )
        if not self.tasks:
            raise ValueError("no tasks defined in tasks.yaml")

    # -- lookups -----------------------------------------------------------
    def task(self, task_id: str) -> TaskSpec:
        for spec in self.tasks:
            if spec.id == task_id:
                return spec
        raise KeyError(f"unknown task id: {task_id}")

    def task_ids(self) -> List[str]:
        return [t.id for t in self.tasks]

    def total_quota(self) -> int:
        return sum(t.quota for t in self.tasks)

    def quota_table(self) -> Dict[str, int]:
        return {t.id: t.quota for t in self.tasks}

    def validate_quota(self, expected_total: Optional[int] = None) -> None:
        """Fail loudly when the quotas no longer add up to the paper's total."""
        total = self.total_quota()
        expected = expected_total or int(self.meta.get("total_effective_pairs", total))
        if total != expected:
            raise ValueError(
                f"task quotas sum to {total}, but the configured total is {expected}; "
                "the manuscript reports 80,000 effective pairs built from Table I"
            )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"PipelineConfig(tasks={self.task_ids()}, total={self.total_quota()})"


def load_tasks(path: Path | str = TASKS_CONFIG) -> PipelineConfig:
    return PipelineConfig(_load_yaml(Path(path)))


def load_filter_rules(path: Path | str = FILTER_CONFIG) -> Dict[str, Any]:
    return _load_yaml(Path(path))


def load_caption_prompts(path: Path | str = CAPTION_PROMPTS) -> Dict[str, Any]:
    return _load_yaml(Path(path))


def load_seeds(path: Path | str = SEEDS_FILE) -> List[Dict[str, Any]]:
    data = _load_yaml(Path(path))
    seeds = data.get("seeds", [])
    if not seeds:
        raise ValueError("no seed captions found in prompts/seeds.yaml")
    return seeds


def load_vqa_prompts(path: Path | str = VQA_PROMPTS) -> Dict[str, Any]:
    return _load_yaml(Path(path))


# ---------------------------------------------------------------- output paths
def data_dir(root: Path | str = REPO_ROOT) -> Path:
    return Path(root) / "data"


def captions_raw_dir(root: Path | str = REPO_ROOT) -> Path:
    return data_dir(root) / "captions" / "raw"


def captions_filtered_dir(root: Path | str = REPO_ROOT) -> Path:
    return data_dir(root) / "captions" / "filtered"


def pairs_dir(root: Path | str = REPO_ROOT) -> Path:
    return data_dir(root) / "pairs"


def manifest_dir(root: Path | str = REPO_ROOT) -> Path:
    return data_dir(root) / "manifest"


def default_ledger(root: Path | str = REPO_ROOT) -> Path:
    """Append-only record of every triple that passed the text filters.

    Keeping one flat ledger makes the image stage resumable: a run that dies
    halfway can restart from the ledger without regenerating captions.
    """
    return data_dir(root) / "ledger.jsonl"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
