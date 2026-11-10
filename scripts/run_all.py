# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--limit", type=int, default=None, help="pairs per task")
    parser.add_argument("--oversample", type=float, default=1.6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def check_configuration() -> int:
    """Validate the configuration without touching any model or image."""
    tasks_cfg = cfg.load_tasks()
    tasks_cfg.validate_quota()

    prompts = cfg.load_caption_prompts()
    seeds = cfg.load_seeds()
    rules = cfg.load_filter_rules()

    missing = [
        spec.prompt_template
        for spec in tasks_cfg.tasks
        if spec.prompt_template not in prompts.get("task_templates", {})
    ]
    if missing:
        print(f"tasks without a prompt template: {missing}")
        return 1

    weak = [
        spec.id
        for spec in tasks_cfg.tasks
        if not [s for s in seeds if spec.prompt_template in s.get("suitable_for", [])]
    ]
    if weak:
        print(f"tasks with no seed caption pointing at a template: {weak}")
        return 1

    text_rules = rules.get("level_1_text", {})
    image_rules = rules.get("level_2_image", {})
    if text_rules.get("bertscore", {}).get("source_vs_edited_min", 0) >= text_rules.get(
        "bertscore", {}
    ).get("source_vs_edited_max", 1):
        print("text filter: source_vs_edited_min must be below source_vs_edited_max")
        return 1
    if image_rules.get("ssim_min", 0) >= image_rules.get("ssim_max", 1):
        print("image filter: ssim_min must be below ssim_max")
        return 1

    region = image_rules.get("edit_region", {})
    if region.get("min_area_fraction", 0) >= region.get("max_area_fraction", 1):
        print("image filter: min_area_fraction must be below max_area_fraction")
        return 1

    quota = tasks_cfg.quota_table()
    print(f"tasks: {len(tasks_cfg.tasks)}, total quota: {sum(quota.values())}")
    for spec in tasks_cfg.tasks:
        seeds_for_task = len(
            [s for s in seeds if spec.prompt_template in s.get("suitable_for", [])]
        )
        print(
            f"  {spec.id:<24} quota {spec.quota:>6}  mode {spec.edit_mode:<10} "
            f"strategy {spec.pair_strategy:<11} seeds {seeds_for_task}"
        )
    print("configuration checks passed")
    return 0


def run(script: str, extra: list[str]) -> int:
    command = [sys.executable, str(SCRIPT_DIR / script)] + extra
    print(f"\n$ {' '.join(command)}")
    return subprocess.call(command)


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return check_configuration()

    task_args = ["--tasks", args.tasks]
    status = 0

    if not args.skip_captions:
        status |= run("01_expand_captions.py", task_args + [
            "--oversample", str(args.oversample), "--workers", str(args.workers)
        ])
        status |= run("02_filter_captions.py", task_args)

    if not args.skip_images:
        extra = task_args + (["--limit", str(args.limit)] if args.limit else [])
        status |= run("03_generate_pairs.py", extra)
        status |= run("04_filter_pairs.py", task_args + (["--no-clip"] if args.no_clip else []))

    status |= run("05_report.py", [])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
