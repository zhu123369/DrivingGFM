# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg
from data_pipeline.dedup import TextDeduper
from data_pipeline.filtering import TextFilter, summarise
from data_pipeline.metrics import BertScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--no-bert", action="store_true", help="skip BERTScore, use token F1")
    parser.add_argument("--dedup-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_cfg = cfg.load_tasks()
    rules = cfg.load_filter_rules()

    bert_enabled = not args.no_bert and bool(
        rules.get("level_1_text", {}).get("bertscore", {}).get("enabled", True)
    )
    bert = BertScorer(
        model=rules.get("level_1_text", {}).get("bertscore", {}).get(
            "model", "distilbert-base-uncased"
        ),
        enabled=bert_enabled,
    )
    if bert_enabled and not bert.available:
        print("BERTScore unavailable; falling back to the flagged lexical proxy")

    task_ids = tasks_cfg.task_ids() if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]

    out_dir = cfg.ensure_dir(cfg.captions_filtered_dir())
    overall = 0
    for task_id in task_ids:
        raw_path = cfg.captions_raw_dir() / f"{task_id}.jsonl"
        records = cfg.read_jsonl(raw_path)
        if not records:
            print(f"[{task_id}] no raw triples; run stage 1 first")
            continue

        filter_engine = TextFilter(
            rules=rules, tasks_cfg=tasks_cfg, bert=bert,
            deduper=TextDeduper(threshold=args.dedup_threshold),
        )
        outcomes = []
        kept = []
        with (out_dir / f"{task_id}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                outcome = filter_engine.check(record)
                outcomes.append(outcome)
                payload = dict(record)
                payload["filters"] = {"text": outcome.as_dict()}
                payload["status"] = "text_pass" if outcome.passed else "text_reject"
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if outcome.passed:
                    kept.append(payload)

        stats = summarise(outcomes)
        quota = tasks_cfg.task(task_id).quota
        print(f"\n[{task_id}] {stats['passed']}/{stats['total']} passed "
              f"({stats['pass_rate']:.1%}), quota {quota}")
        for reason, count in list(stats["reasons"].items())[:8]:
            print(f"    {reason}: {count}")
        overall += len(kept)

    print(f"\ntriples surviving the text rules: {overall}")
    print(f"records live under {cfg.captions_filtered_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
