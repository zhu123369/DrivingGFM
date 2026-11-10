# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg


# Targets reported in Section IV-B: 0.92 for source vs edited, 0.81 for
# instruction vs edited. The lexical fallback sits on a different scale.
BERT_TARGETS = (0.92, 0.81)
F1_TARGETS = (0.93, 0.35)


def text_quality_score(metrics: Dict[str, Any]) -> float:
    """Rank triples inside a subtask bucket.

    Distance to the reported means, not "highest similarity wins": a triple whose
    captions are 0.99 alike carries almost no edit, and one at 0.60 is a different
    scene. Closeness to 0.92 is what the released dataset actually looks like.
    """
    source_vs_edited = float(metrics.get("bertscore_source_edited", 0.0))
    instruction_vs_edited = float(metrics.get("bertscore_instruction_edited", 0.0))
    duplicate = float(metrics.get("nearest_duplicate_jaccard", 0.0))

    if metrics.get("metric") == "token_f1":
        source_target, instruction_target = F1_TARGETS
    else:
        source_target, instruction_target = BERT_TARGETS

    source_term = 1.0 - min(1.0, abs(source_vs_edited - source_target) / 0.20)
    instruction_term = 1.0 - min(1.0, abs(instruction_vs_edited - instruction_target) / 0.30)
    return 0.5 * source_term + 0.3 * instruction_term + 0.2 * (1.0 - duplicate)


def bucket_of(record: Dict[str, Any]) -> str:
    subtask = str(record.get("subtask", "")).strip().lower()
    return subtask or "unspecified"


def select_task(records: List[Dict[str, Any]], quota: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pick ``quota`` triples, round-robin across subtask buckets."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        record["_score"] = round(text_quality_score(record.get("filters", {}).get("text", {}).get("metrics", {})), 4)
        buckets.setdefault(bucket_of(record), []).append(record)

    for bucket in buckets.values():
        bucket.sort(key=lambda r: r["_score"], reverse=True)

    # Bigger buckets first, so the small ones are not exhausted before they get
    # their turn; the ordering only affects which triple is taken earlier.
    order = sorted(buckets.keys(), key=lambda name: len(buckets[name]), reverse=True)

    chosen: List[Dict[str, Any]] = []
    depth = 0
    while len(chosen) < quota:
        added = False
        for name in order:
            bucket = buckets[name]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                added = True
                if len(chosen) >= quota:
                    break
        if not added:  # every bucket exhausted before the quota was filled
            break
        depth += 1

    return chosen, {
        "candidates": len(records),
        "selected": len(chosen),
        "subtasks": len(buckets),
        "depth_used": depth,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--require-full", action="store_true",
                        help="exit non-zero when a task cannot reach its quota")
    args = parser.parse_args()

    tasks_cfg = cfg.load_tasks()
    tasks_cfg.validate_quota()
    task_ids = tasks_cfg.task_ids() if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]

    out_dir = cfg.ensure_dir(cfg.manifest_dir())
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"mode": "text_only", "tasks": {}, "total": 0, "quota_total": 0}
    shortfalls: List[str] = []

    for task_id in task_ids:
        spec = tasks_cfg.task(task_id)
        records = [
            r for r in cfg.read_jsonl(cfg.captions_filtered_dir() / f"{task_id}.jsonl")
            if r.get("status") == "text_pass"
        ]
        if not records:
            shortfalls.append(f"{task_id}: no text-passing triples (run stages 1-2)")
            print(f"[{task_id}] 0 candidates, quota {spec.quota}")
            continue

        chosen, stats = select_task(records, spec.quota)
        rows = []
        for record in chosen:
            record.pop("_score", None)
            record["status"] = "selected"
            rows.append({
                "pair_id": record.get("pair_key"),
                "task": task_id,
                "subtask": record.get("subtask", ""),
                "source_caption": record.get("source_caption"),
                "instruction": record.get("instruction"),
                "edited_caption": record.get("edited_caption"),
                "edited_region": record.get("edited_region", ""),
                "image_source": None,
                "image_edited": None,
                "filters": record.get("filters", {}),
                "gen": record.get("gen", {}),
                "status": "selected",
            })

        with (out_dir / f"{task_id}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary["tasks"][task_id] = dict(stats, quota=spec.quota, filled=len(rows) >= spec.quota)
        summary["total"] += len(rows)
        summary["quota_total"] += spec.quota
        all_rows.extend(rows)

        flag = "" if len(rows) >= spec.quota else f"  SHORT by {spec.quota - len(rows)}"
        print(f"[{task_id}] {len(rows)}/{spec.quota} selected from {stats['candidates']} "
              f"candidates over {stats['subtasks']} subtasks{flag}")
        if len(rows) < spec.quota:
            shortfalls.append(f"{task_id}: {len(rows)}/{spec.quota}")

    with (out_dir / "all.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["complete"] = not shortfalls
    summary["shortfalls"] = shortfalls
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\ntotal selected: {summary['total']} / {summary['quota_total']}")
    print(f"manifest: {out_dir / 'all.jsonl'}")
    if shortfalls:
        print("shortfalls (generate more triples for these tasks):")
        for item in shortfalls:
            print(f"  - {item}")
        return 1 if args.require_full else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
