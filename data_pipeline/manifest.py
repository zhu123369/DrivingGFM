# -*- coding: utf-8 -*-
"""Assemble the released dataset manifest.

Layout produced::

    data/manifest/all.jsonl            every selected pair, task order preserved
    data/manifest/<task>.jsonl         per-task view, exactly `quota` lines
    data/manifest/summary.json         counts and per-task statistics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as cfg


MANIFEST_FIELDS = (
    "pair_id",
    "task",
    "subtask",
    "source_caption",
    "instruction",
    "edited_caption",
    "edited_region",
    "image_source",
    "image_edited",
    "gen",
    "filters",
    "status",
)


def manifest_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Project a ledger record onto the stable manifest schema."""
    entry = {field: record.get(field) for field in MANIFEST_FIELDS}
    entry["task"] = record.get("task")
    entry["filters"] = record.get("filters", {})
    return entry


def build_manifest(
    root: Path | str = cfg.REPO_ROOT,
    tasks_cfg: Optional[cfg.PipelineConfig] = None,
    quotas: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Read the per-task pair ledgers, keep exactly the selected pairs."""
    root = Path(root)
    tasks_cfg = tasks_cfg or cfg.load_tasks()
    quotas = quotas or tasks_cfg.quota_table()

    out_dir = cfg.ensure_dir(cfg.manifest_dir(root))
    all_entries: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"tasks": {}, "total": 0, "quota_total": sum(quotas.values())}

    for spec in tasks_cfg.tasks:
        ledger = cfg.pairs_dir(root) / f"{spec.id}.jsonl"
        records = cfg.read_jsonl(ledger)
        selected = [
            record for record in records
            if record.get("status") == "selected"
        ]
        if len(selected) > quotas[spec.id]:
            selected = selected[: quotas[spec.id]]

        entries = [manifest_entry(record) for record in selected]
        with (out_dir / f"{spec.id}.jsonl").open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        stats = _task_stats(entries)
        stats["quota"] = quotas[spec.id]
        stats["generated"] = len(records)
        stats["filled"] = len(entries) >= quotas[spec.id]
        summary["tasks"][spec.id] = stats
        summary["total"] += len(entries)
        all_entries.extend(entries)

    with (out_dir / "all.jsonl").open("w", encoding="utf-8") as handle:
        for entry in all_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary["complete"] = all(
        summary["tasks"][task_id]["filled"] for task_id in quotas
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary


def _task_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {"count": 0, "mean_ssim": None, "mean_clip_d": None}

    ssim_values = [
        entry["filters"].get("metrics", {}).get("ssim")
        for entry in entries
        if entry.get("filters")
    ]
    clip_values = [
        entry["filters"].get("metrics", {}).get("clip_d")
        for entry in entries
        if entry.get("filters")
    ]
    ssim_clean = [float(v) for v in ssim_values if v is not None]
    clip_clean = [float(v) for v in clip_values if v is not None]

    return {
        "count": len(entries),
        "mean_ssim": round(sum(ssim_clean) / len(ssim_clean), 4) if ssim_clean else None,
        "mean_clip_d": round(sum(clip_clean) / len(clip_clean), 4) if clip_clean else None,
        "subtasks": sorted({entry.get("subtask", "unspecified") for entry in entries}),
    }


def verify_quota_table(summary: Dict[str, Any], tasks_cfg: cfg.PipelineConfig) -> List[str]:
    """Return a list of per-task mismatch messages; empty means the table holds."""
    problems: List[str] = []
    for spec in tasks_cfg.tasks:
        stats = summary["tasks"].get(spec.id, {})
        count = int(stats.get("count", 0))
        if count != spec.quota:
            problems.append(f"{spec.id}: {count} pairs selected, Table I says {spec.quota}")
    if summary.get("total") != tasks_cfg.total_quota():
        problems.append(
            f"total {summary.get('total')} does not match the paper's {tasks_cfg.total_quota()}"
        )
    return problems
