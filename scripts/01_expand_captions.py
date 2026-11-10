# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg
from data_pipeline.caption_expand import expand_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="comma separated task ids, or 'all'")
    parser.add_argument("--oversample", type=float, default=1.6, help="raw triples per quota")
    parser.add_argument("--workers", type=int, default=8, help="concurrent LLM calls")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tasks-config", default=str(cfg.TASKS_CONFIG))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_cfg = cfg.load_tasks(args.tasks_config)
    tasks_cfg.validate_quota()

    if args.tasks == "all":
        task_ids = tasks_cfg.task_ids()
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    total_written = 0
    for task_id in task_ids:
        spec = tasks_cfg.task(task_id)
        target = math.ceil(spec.quota * args.oversample)
        print(f"\n=== {task_id}: target {target} raw triples for a quota of {spec.quota} ===")
        written = expand_task(
            task_id,
            target=target,
            workers=args.workers,
            seed=args.seed,
            tasks_cfg=tasks_cfg,
        )
        total_written += written

    print(f"\nraw triples written in this run: {total_written}")
    print(f"records live under {cfg.captions_raw_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
