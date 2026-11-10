# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg
from data_pipeline.dedup import ImageDeduper
from data_pipeline.filtering import ImageFilter, quality_score, summarise, select_for_quota
from data_pipeline.metrics import ClipScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="pairs per CLIP batch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_cfg = cfg.load_tasks()
    rules = cfg.load_filter_rules()

    from PIL import Image

    clip_rules = rules.get("level_2_image", {}).get("clip_directional", {})
    clip = None
    if not args.no_clip and clip_rules.get("enabled", True):
        clip = ClipScorer(
            model_id=clip_rules.get("clip_model", "openai/clip-vit-large-patch14"),
            device=args.device,
            enabled=True,
        )
        if not clip.available:
            print("CLIP unavailable; instruction-alignment rule will be skipped")

    task_ids = tasks_cfg.task_ids() if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]

    for task_id in task_ids:
        spec = tasks_cfg.task(task_id)
        ledger_path = cfg.pairs_dir() / f"{task_id}.jsonl"
        records = cfg.read_jsonl(ledger_path)
        if not records:
            print(f"[{task_id}] no pairs; run stage 3 first")
            continue

        engine = ImageFilter(rules=rules, clip=clip, deduper=ImageDeduper(
            hash_size=int(rules["level_2_image"]["duplicate_detection"]["hash_size"]),
            max_hamming=int(rules["level_2_image"]["duplicate_detection"]["max_hamming"]),
            max_edits_per_source=int(
                rules["level_2_image"]["duplicate_detection"]["max_edits_per_source"]
            ),
        ))

        outcomes = []
        pending = [r for r in records if r.get("status") == "generated"]

        batch: list = []
        for record in pending + [None]:
            if record is not None:
                batch.append(record)
            if len(batch) == args.batch_size or (record is None and batch):
                images = []
                for item in batch:
                    try:
                        images.append((
                            Image.open(cfg.REPO_ROOT / item["image_source"]).convert("RGB"),
                            Image.open(cfg.REPO_ROOT / item["image_edited"]).convert("RGB"),
                        ))
                    except (FileNotFoundError, OSError):
                        images.append((None, None))

                clip_scores = None
                if clip is not None and clip.available:
                    valid = [i for i, (src, _) in enumerate(images) if src is not None]
                    if valid:
                        try:
                            clip_scores = clip.directional_scores(
                                [images[i][0] for i in valid],
                                [images[i][1] for i in valid],
                                [batch[i]["instruction"] for i in valid],
                            )
                        except Exception as exc:
                            print(f"[{task_id}] CLIP batch failed: {type(exc).__name__}: {exc}")

                for index, item in enumerate(batch):
                    source, edited = images[index]
                    if source is None:
                        item["filters"] = {"image": {"passed": False,
                                                     "violations": ["image.missing_file"],
                                                     "metrics": {}}}
                        item["status"] = "image_reject"
                        continue
                    score = None
                    if clip_scores is not None:
                        valid_indices = [i for i, (src, _) in enumerate(images) if src is not None]
                        if index in valid_indices:
                            score = clip_scores[valid_indices.index(index)]
                    outcome = engine.check(source, edited, item.get("instruction", ""), clip_d=score)
                    outcomes.append(outcome)
                    item["filters"] = {"image": outcome.as_dict()}
                    item["status"] = "image_pass" if outcome.passed else "image_reject"

                batch = []

        # Select exactly the quota, best first.
        passed = [r for r in records if r.get("status") == "image_pass"]
        for record in passed:
            metrics = record.get("filters", {}).get("image", {}).get("metrics", {})
            record["quality_score"] = round(quality_score(metrics), 4)
        select_for_quota(passed, spec.quota)

        with ledger_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        stats = summarise(outcomes)
        selected = sum(1 for r in records if r.get("status") == "selected")
        print(f"\n[{task_id}] {stats['passed']}/{stats['total']} image checks passed, "
              f"selected {selected}/{spec.quota}")
        for reason, count in list(stats["reasons"].items())[:8]:
            print(f"    {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
