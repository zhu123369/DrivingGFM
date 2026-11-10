# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg
from data_pipeline.pair_generation import (
    DEFAULT_MODEL,
    PairGenerator,
    SamplingSettings,
    generate_task_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="pairs per task, default quota")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=77)
    parser.add_argument("--replace-end", type=float, default=0.5,
                        help="fraction of the schedule over which attention is exchanged")
    parser.add_argument("--store-blocks", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--strategy", default=None, choices=["p2p", "degradation"],
                        help="override the per-task strategy in configs/tasks.yaml")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_cfg = cfg.load_tasks()

    settings = SamplingSettings(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.size,
        width=args.size,
        max_sequence_length=args.max_seq,
        replace_end=args.replace_end,
        store_blocks=tuple(int(v) for v in args.store_blocks.split(",") if v.strip()),
        device=args.device,
        dtype=args.dtype,
    )
    generator = PairGenerator(args.model, settings)

    task_ids = tasks_cfg.task_ids() if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]

    total = 0
    for task_id in task_ids:
        spec = tasks_cfg.task(task_id)
        records = [
            record
            for record in cfg.read_jsonl(cfg.captions_filtered_dir() / f"{task_id}.jsonl")
            if record.get("status") == "text_pass"
        ]
        if not records:
            print(f"[{task_id}] no text-passing triples; run stage 2 first")
            continue

        limit = args.limit if args.limit is not None else spec.quota
        for record in records[:limit]:
            record["edit_mode"] = spec.edit_mode
            record["pair_strategy"] = args.strategy or spec.pair_strategy
            record.setdefault("seed", args.seed)

        print(f"\n=== {task_id}: rendering {len(records[:limit])} pairs "
              f"({spec.edit_mode}, {args.strategy or spec.pair_strategy}) ===")
        total += generate_task_pairs(task_id, records[:limit], generator)

    print(f"\npairs rendered in this run: {total}")
    print(f"images live under {cfg.pairs_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
