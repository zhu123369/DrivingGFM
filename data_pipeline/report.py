# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import config as cfg
from .manifest import build_manifest
from .metrics import BertScorer, ClipScorer, fid_to_reference, ssim


def _bootstrap_ci(values: Sequence[float], iterations: int = 1000, confidence: float = 0.95) -> Optional[List[float]]:
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(2026)
    means = [float(array[rng.integers(0, len(array), len(array))].mean()) for _ in range(iterations)]
    alpha = (1.0 - confidence) / 2.0
    return [round(float(np.quantile(means, alpha)), 4), round(float(np.quantile(means, 1 - alpha)), 4)]


def _load_image(root: Path, relative: str):
    from PIL import Image

    return Image.open(root / relative).convert("RGB")


def dataset_quality_report(
    root: Path | str = cfg.REPO_ROOT,
    sample_size: int = 10000,
    *,
    rules: Optional[Dict[str, Any]] = None,
    with_clip: bool = False,
    with_bert: bool = True,
    fid_references: Optional[Dict[str, str]] = None,
    output: Optional[Path] = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Build and write the quality report, returning it as a dict."""
    root = Path(root)
    rules = rules or cfg.load_filter_rules()
    report_cfg = rules.get("level_3_dataset", {})
    sample_size = int(sample_size or report_cfg.get("evaluation_sample", 10000))

    manifest_dir = cfg.manifest_dir(root)
    summary_path = manifest_dir / "summary.json"
    if not summary_path.exists():
        build_manifest(root)
    records = cfg.read_jsonl(manifest_dir / "all.jsonl")

    rng = random.Random(seed)
    sample = records if len(records) <= sample_size else rng.sample(records, sample_size)

    report: Dict[str, Any] = {
        "dataset": {
            "total_pairs": len(records),
            "tasks": json.loads(summary_path.read_text(encoding="utf-8"))["tasks"] if summary_path.exists() else {},
            "evaluation_sample": len(sample),
        },
        "text": {},
        "structure": {},
        "alignment": {},
        "distribution": {},
    }

    # -- text -------------------------------------------------------------
    bert = BertScorer(enabled=with_bert)
    if sample and with_bert:
        source_edited: List[float] = []
        instruction_edited: List[float] = []
        for record in sample:
            value, metric = bert.score(record["source_caption"], record["edited_caption"])
            source_edited.append(value)
            value2, _ = bert.score(record["instruction"], record["edited_caption"])
            instruction_edited.append(value2)
        report["text"] = {
            "computed": True,
            "metric": "bertscore_f1" if bert.available else "token_f1_fallback",
            "source_vs_edited_mean": round(float(np.mean(source_edited)), 4),
            "instruction_vs_edited_mean": round(float(np.mean(instruction_edited)), 4),
            "source_vs_edited_std": round(float(np.std(source_edited)), 4),
        }
    else:
        report["text"] = {"computed": False, "reason": "BERTScore disabled or no pairs in the manifest"}

    # -- structure --------------------------------------------------------
    stored_ssim = [
        float(record["filters"]["metrics"]["ssim"])
        for record in sample
        if record.get("filters", {}).get("metrics", {}).get("ssim") is not None
    ]
    if stored_ssim:
        report["structure"] = {
            "computed": True,
            "source": "filter metrics (recorded at generation time)",
            "ssim_mean": round(float(np.mean(stored_ssim)), 4),
            "ssim_p05": round(float(np.quantile(stored_ssim, 0.05)), 4),
        }
    else:
        recomputed = []
        for record in sample[: min(len(sample), 2000)]:
            try:
                recomputed.append(
                    ssim(_load_image(root, record["image_source"]), _load_image(root, record["image_edited"]))
                )
            except (FileNotFoundError, OSError):
                continue
        report["structure"] = {
            "computed": bool(recomputed),
            "source": "recomputed from images",
            "ssim_mean": round(float(np.mean(recomputed)), 4) if recomputed else None,
        }

    # -- instruction alignment -------------------------------------------
    stored_clip = [
        float(record["filters"]["metrics"]["clip_d"])
        for record in sample
        if record.get("filters", {}).get("metrics", {}).get("clip_d") is not None
    ]
    alignment: Dict[str, Any] = {}
    if stored_clip:
        alignment.update(
            {
                "computed": True,
                "clip_d_mean": round(float(np.mean(stored_clip)), 4),
                "clip_d_ci": _bootstrap_ci(
                    stored_clip,
                    iterations=int(report_cfg.get("bootstrap_iterations", 1000)),
                    confidence=float(report_cfg.get("confidence", 0.95)),
                ),
                "ci_source": "filter metrics",
            }
        )

    if with_clip and sample:
        clip = ClipScorer(device="cuda", enabled=True)
        if clip.available:
            limit = min(len(sample), 500)
            pairs = sample[:limit]
            sources = [_load_image(root, record["image_source"]) for record in pairs]
            edits = [_load_image(root, record["image_edited"]) for record in pairs]
            instructions = [record["instruction"] for record in pairs]

            paired = clip.directional_scores(sources, edits, instructions)
            shuffled_instructions = list(instructions)
            rng.shuffle(shuffled_instructions)
            baseline = clip.directional_scores(sources, edits, shuffled_instructions)

            alignment = {
                "computed": True,
                "clip_d_mean": round(float(np.mean(paired)), 4),
                "clip_d_ci": _bootstrap_ci(
                    paired,
                    iterations=int(report_cfg.get("bootstrap_iterations", 1000)),
                    confidence=float(report_cfg.get("confidence", 0.95)),
                ),
                "random_baseline_mean": round(float(np.mean(baseline)), 4),
                "recomputed_pairs": len(paired),
                "model": clip.model_id,
            }
        else:
            alignment.setdefault("computed", False)
            alignment["clip_note"] = "CLIP unavailable, stored filter values used"

    if not alignment:
        alignment = {"computed": False, "reason": "no CLIP-D values recorded and CLIP disabled"}
    report["alignment"] = alignment

    # -- distribution -----------------------------------------------------
    distribution: Dict[str, Any] = {"computed": False}
    if fid_references:
        fids = {}
        for name, path in fid_references.items():
            value = fid_to_reference(str(cfg.pairs_dir(root)), str(path))
            fids[name] = value
        distribution = {
            "computed": any(value is not None for value in fids.values()),
            "fid": fids,
            "note": "FID is a distribution check; it is reported, never used to drop a sample",
        }
    report["distribution"] = distribution

    out_path = Path(output) if output else (root / report_cfg.get("report_path", "reports/dataset_quality.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    """A short human-readable summary of the report, for the release notes."""
    lines = ["# DrivingGFM dataset quality report", ""]
    dataset = report.get("dataset", {})
    lines.append(f"- pairs in manifest: **{dataset.get('total_pairs')}**")
    lines.append(f"- pairs evaluated: {dataset.get('evaluation_sample')}")
    lines.append("")

    text = report.get("text", {})
    if text.get("computed"):
        lines.append("## Text")
        lines.append(f"- metric: `{text.get('metric')}`")
        lines.append(f"- source vs edited caption: {text.get('source_vs_edited_mean')}")
        lines.append(f"- instruction vs edited caption: {text.get('instruction_vs_edited_mean')}")
        lines.append("")

    structure = report.get("structure", {})
    if structure.get("computed"):
        lines.append("## Structure")
        lines.append(f"- mean SSIM: {structure.get('ssim_mean')} ({structure.get('source')})")
        lines.append("")

    alignment = report.get("alignment", {})
    if alignment.get("computed"):
        lines.append("## Instruction alignment")
        lines.append(f"- mean CLIP-D: {alignment.get('clip_d_mean')}")
        if alignment.get("clip_d_ci"):
            lines.append(f"- 95% CI: {alignment.get('clip_d_ci')}")
        if alignment.get("random_baseline_mean") is not None:
            lines.append(f"- shuffled baseline: {alignment.get('random_baseline_mean')}")
        lines.append("")

    distribution = report.get("distribution", {})
    if distribution.get("computed"):
        lines.append("## Distribution")
        for name, value in (distribution.get("fid") or {}).items():
            lines.append(f"- FID vs {name}: {value}")
        lines.append("")

    tasks = dataset.get("tasks") or {}
    if tasks:
        lines.append("## Per task")
        lines.append("| task | pairs | quota | mean SSIM | mean CLIP-D |")
        lines.append("|---|---|---|---|---|")
        for task_id, stats in tasks.items():
            lines.append(
                f"| {task_id} | {stats.get('count')} | {stats.get('quota')} | "
                f"{stats.get('mean_ssim')} | {stats.get('mean_clip_d')} |"
            )
    return "\n".join(lines) + "\n"
