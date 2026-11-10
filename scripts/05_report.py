# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config as cfg
from data_pipeline.manifest import build_manifest, verify_quota_table
from data_pipeline.report import dataset_quality_report, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-bert", action="store_true", default=True)
    parser.add_argument("--no-bert", dest="with_bert", action="store_false")
    parser.add_argument("--with-clip", action="store_true",
                        help="recompute CLIP-D, its shuffled baseline and the CI")
    parser.add_argument("--sample", type=int, default=0, help="pairs to evaluate, default 10000")
    parser.add_argument("--fid-kitti", default=None)
    parser.add_argument("--fid-cityscapes", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_cfg = cfg.load_tasks()

    summary = build_manifest(tasks_cfg=tasks_cfg)
    print(f"manifest: {summary['total']} pairs across {len(summary['tasks'])} tasks")

    problems = verify_quota_table(summary, tasks_cfg)
    if problems:
        print("per-task counts do not match Table I:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("every task matches its Table I quota")

    references = {}
    if args.fid_kitti:
        references["KITTI"] = args.fid_kitti
    if args.fid_cityscapes:
        references["Cityscapes"] = args.fid_cityscapes

    report = dataset_quality_report(
        sample_size=args.sample or 10000,
        with_clip=args.with_clip,
        with_bert=args.with_bert,
        fid_references=references or None,
    )
    markdown = render_markdown(report)
    markdown_path = Path(report["report_path"]).with_suffix(".md")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"report written to {report['report_path']} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
