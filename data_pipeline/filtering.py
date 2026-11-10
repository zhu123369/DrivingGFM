# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config as cfg
from .dedup import ImageDeduper, TextDeduper, normalize
from .metrics import (
    BertScorer,
    ClipScorer,
    channel_shift,
    edit_region_stats,
    photometric_stats,
    ssim,
    token_f1,
)

# Imperative verbs that may open an editing instruction. A sentence that starts
# with anything else is usually a description ("the sign is blue") rather than
# an instruction, and Prompt-to-Prompt has nothing to condition on.
EDIT_VERBS = (
    "mark", "highlight", "outline", "draw", "recolour", "recolor", "colour",
    "color", "remove", "erase", "delete", "clean", "brighten", "sharpen",
    "clear", "make", "add", "recover", "raise", "restore", "change",
    "improve", "dehaze", "defog", "desnow", "derain", "lift",
)

TARGET_PATTERN = re.compile(r"\b(?P<count>\d+|one|two|three|four|several)\b")


@dataclass
class FilterOutcome:
    passed: bool
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": list(self.violations),
            "metrics": dict(self.metrics),
        }

    def reject(self, reason: str) -> "FilterOutcome":
        self.passed = False
        if reason not in self.violations:
            self.violations.append(reason)
        return self


# --------------------------------------------------------------------- text stage
class TextFilter:
    """Applies ``level_1_text`` plus the per-task consistency rules."""

    def __init__(
        self,
        rules: Dict[str, Any],
        tasks_cfg: Optional[cfg.PipelineConfig] = None,
        bert: Optional[BertScorer] = None,
        deduper: Optional[TextDeduper] = None,
    ):
        self.rules = rules
        self.text_rules = rules.get("level_1_text", {})
        self.tasks_cfg = tasks_cfg or cfg.load_tasks()
        self.bert = bert if bert is not None else BertScorer(
            model=self.text_rules.get("bertscore", {}).get("model", "distilbert-base-uncased"),
            enabled=bool(self.text_rules.get("bertscore", {}).get("enabled", True)),
        )
        self.deduper = deduper or TextDeduper()
        self._anchor_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.text_rules.get("edit_locality", {}).get("anchor_patterns", [])
        ]
        self._deny_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.text_rules.get("safety_guards", {}).get("deny_patterns", [])
        ]

    # -- public -----------------------------------------------------------
    def check(self, record: Dict[str, Any]) -> FilterOutcome:
        outcome = FilterOutcome(passed=True)
        self._check_shape(record, outcome)
        if not outcome.passed:
            return outcome

        self._check_locality(record, outcome)
        self._check_task_consistency(record, outcome)
        self._check_safety(record, outcome)
        self._check_similarity(record, outcome)

        if outcome.passed and self.text_rules.get("edit_locality", {}).get("reject_multi_edit", True):
            self._check_single_edit(record, outcome)

        if outcome.passed:
            accepted, similarity = self.deduper.accept(record)
            outcome.metrics["nearest_duplicate_jaccard"] = round(similarity, 4)
            if not accepted:
                outcome.reject("text.duplicate")

        return outcome

    # -- checks -----------------------------------------------------------
    def _check_shape(self, record: Dict[str, Any], outcome: FilterOutcome) -> None:
        for field_name in ("source_caption", "instruction", "edited_caption"):
            value = str(record.get(field_name, "")).strip()
            if not value:
                outcome.reject(f"text.empty_{field_name}")
                return

        instruction_words = _words(record["instruction"])
        caption_words = _words(record["source_caption"])
        outcome.metrics["instruction_words"] = len(instruction_words)
        outcome.metrics["caption_words"] = len(caption_words)

        if len(instruction_words) < 3:
            outcome.reject("text.instruction_too_short")
        if len(instruction_words) > 25:
            outcome.reject("text.instruction_too_long")
        min_words = int(self.text_rules.get("min_words", 8))
        max_words = int(self.text_rules.get("max_words", 55))
        if not (min_words <= len(caption_words) <= max_words):
            outcome.reject("text.caption_length")
        if len(str(record["source_caption"])) < int(self.text_rules.get("min_chars", 40)):
            outcome.reject("text.caption_too_short")

        sentences = [s for s in re.split(r"[.!?]+", record["source_caption"]) if s.strip()]
        if len(sentences) > int(self.text_rules.get("max_sentences", 3)):
            outcome.reject("text.too_many_sentences")

        if normalize(record["source_caption"]) == normalize(record["edited_caption"]):
            outcome.reject("text.no_change")
        if normalize(record["instruction"]) == normalize(record["edited_caption"]):
            outcome.reject("text.instruction_is_caption")

        first = instruction_words[0].lower() if instruction_words else ""
        if first not in EDIT_VERBS:
            outcome.reject("text.not_imperative")

    def _check_locality(self, record: Dict[str, Any], outcome: FilterOutcome) -> None:
        locality = self.text_rules.get("edit_locality", {})
        instruction = record["instruction"]

        if locality.get("require_anchor", True) and self._anchor_patterns:
            if not any(pattern.search(instruction) for pattern in self._anchor_patterns):
                outcome.reject("text.no_anchor")

        max_drop = int(locality.get("preserved_nouns_max_drop", 1))
        dropped = 0
        for noun in locality.get("preserved_nouns", []):
            source_count = _count_term(record["source_caption"], noun)
            edited_count = _count_term(record["edited_caption"], noun)
            if source_count > 0 and edited_count - source_count < -max_drop:
                dropped += 1
        outcome.metrics["preserved_nouns_dropped"] = dropped
        if dropped > max_drop:
            outcome.reject("text.scene_rewritten")

    def _check_task_consistency(self, record: Dict[str, Any], outcome: FilterOutcome) -> None:
        rules = self.text_rules.get("task_consistency", {})
        if not rules.get("enforce_condition_terms", True):
            return
        task = record.get("task")
        source = record["source_caption"]
        instruction = record["instruction"]

        terms: Sequence[str] = []
        if task in rules.get("condition_terms", {}):
            terms = rules["condition_terms"][task]
        elif task in rules.get("detection_classes", {}):
            terms = rules["detection_classes"][task]

        if not terms:
            return
        requirement = rules.get("require_term_in_source", {})
        if requirement.get(task, True) and not any(
            term.lower() in source.lower() for term in terms
        ):
            outcome.reject("text.condition_absent_from_caption")
            return
        if not any(term.lower() in instruction.lower() for term in terms):
            outcome.reject("text.condition_absent_from_instruction")

    def _check_safety(self, record: Dict[str, Any], outcome: FilterOutcome) -> None:
        guards = self.text_rules.get("safety_guards", {})
        if not guards.get("enabled", True):
            return
        haystack = f"{record['instruction']} {record['edited_caption']}"
        for pattern in self._deny_patterns:
            if pattern.search(haystack):
                outcome.reject(f"safety.denied:{pattern.pattern}")
                return

        max_targets = int(guards.get("max_targets_per_instruction", 1))
        targets = TARGET_PATTERN.findall(record["instruction"])
        if len(targets) > max_targets:
            outcome.reject("safety.multiple_targets")

    def _check_single_edit(self, record: Dict[str, Any], outcome: FilterOutcome) -> None:
        connectives = self.text_rules.get("edit_locality", {}).get("forbidden_connectives", [])
        instruction = record["instruction"].lower()
        for connective in connectives:
            if connective in instruction:
                outcome.reject(f"text.multi_edit:{connective}")
                return

    def _check_similarity(self, record: Dict[str, Any], outcome: FilterOutcome) -> None:
        bert_rules = self.text_rules.get("bertscore", {})
        if not bert_rules.get("enabled", True):
            return

        source_edited, metric = self.bert.score(record["source_caption"], record["edited_caption"])
        instruction_edited, _ = self.bert.score(record["instruction"], record["edited_caption"])
        outcome.metrics["metric"] = metric
        outcome.metrics["bertscore_source_edited"] = round(source_edited, 4)
        outcome.metrics["bertscore_instruction_edited"] = round(instruction_edited, 4)

        if metric == "token_f1":
            # Lexical F1 sits on a different scale; using the semantic thresholds
            # here would reject almost everything.
            fallback = bert_rules.get("fallback_when_unavailable", {})
            low = float(fallback.get("source_vs_edited_min", 0.60))
            high = float(fallback.get("source_vs_edited_max", 0.98))
            instruction_low = float(fallback.get("instruction_vs_edited_min", 0.30))
        else:
            low = float(bert_rules.get("source_vs_edited_min", 0.88))
            high = float(bert_rules.get("source_vs_edited_max", 0.98))
            instruction_low = float(bert_rules.get("instruction_vs_edited_min", 0.72))

        if source_edited < low:
            outcome.reject("text.deviation_too_large")
        elif source_edited > high:
            outcome.reject("text.deviation_too_small")
        if instruction_edited < instruction_low:
            outcome.reject("text.instruction_mismatch")


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text))


def _count_term(text: str, term: str) -> int:
    return len(re.findall(rf"\b{re.escape(term)}\b", str(text), flags=re.IGNORECASE))


# -------------------------------------------------------------------- image stage
class ImageFilter:
    """Applies ``level_2_image`` to rendered pairs."""

    def __init__(
        self,
        rules: Dict[str, Any],
        clip: Optional[ClipScorer] = None,
        deduper: Optional[ImageDeduper] = None,
    ):
        self.rules = rules
        self.image_rules = rules.get("level_2_image", {})
        self.clip = clip
        self.deduper = deduper or ImageDeduper(
            hash_size=int(self.image_rules.get("duplicate_detection", {}).get("hash_size", 8)),
            max_hamming=int(self.image_rules.get("duplicate_detection", {}).get("max_hamming", 6)),
            max_edits_per_source=int(
                self.image_rules.get("duplicate_detection", {}).get("max_edits_per_source", 1)
            ),
        )
        self._bert = BertScorer(enabled=False)

    def check(
        self,
        source_image: Any,
        edited_image: Any,
        instruction: str = "",
        clip_d: Optional[float] = None,
    ) -> FilterOutcome:
        outcome = FilterOutcome(passed=True)
        rules = self.image_rules

        try:
            score = ssim(source_image, edited_image)
        except ValueError:
            return outcome.reject("image.size_mismatch")
        outcome.metrics["ssim"] = round(score, 4)
        if score < float(rules.get("ssim_min", 0.80)):
            outcome.reject("image.structure_drift")
        elif score > float(rules.get("ssim_max", 0.99)):
            outcome.reject("image.no_change")

        region = edit_region_stats(
            source_image,
            edited_image,
            pixel_threshold=float(rules.get("edit_region", {}).get("pixel_threshold", 0.06)),
            opening_kernel=int(rules.get("edit_region", {}).get("opening_kernel", 3)),
        )
        outcome.metrics["edit_area_fraction"] = round(region["area_fraction"], 5)
        outcome.metrics["largest_component_share"] = round(region["largest_component_share"], 4)
        outcome.metrics["edit_bbox"] = region["bbox"]

        region_rules = rules.get("edit_region", {})
        if region["area_fraction"] < float(region_rules.get("min_area_fraction", 0.002)):
            outcome.reject("image.edit_too_small")
        elif region["area_fraction"] > float(region_rules.get("max_area_fraction", 0.35)):
            outcome.reject("image.edit_too_large")
        if region["largest_component_share"] < float(
            region_rules.get("min_largest_component_share", 0.40)
        ):
            outcome.reject("image.edit_scattered")

        stats_source = photometric_stats(source_image)
        stats_edit = photometric_stats(edited_image)
        outcome.metrics["edit_mean_luma"] = round(stats_edit["mean_luma"], 4)
        outcome.metrics["edit_std_luma"] = round(stats_edit["std_luma"], 4)
        photo = rules.get("photometric", {})
        if stats_edit["mean_luma"] < float(photo.get("min_mean_luma", 0.05)):
            outcome.reject("image.too_dark")
        elif stats_edit["mean_luma"] > float(photo.get("max_mean_luma", 0.97)):
            outcome.reject("image.blown_out")
        if stats_edit["std_luma"] < float(photo.get("min_std_luma", 0.02)):
            outcome.reject("image.flat")
        shift = channel_shift(source_image, edited_image)
        outcome.metrics["mean_channel_shift"] = round(shift, 4)
        if shift > float(photo.get("max_mean_channel_shift", 0.35)):
            outcome.reject("image.colour_drift")

        clip_rules = rules.get("clip_directional", {})
        if clip_rules.get("enabled", True) and clip_d is not None:
            outcome.metrics["clip_d"] = round(float(clip_d), 4)
            if float(clip_d) < float(clip_rules.get("min_score", 0.08)):
                outcome.reject("image.instruction_misaligned")

        # Only pairs that are otherwise acceptable enter the duplicate registry:
        # a rejected pair must not make its own frame unavailable to a later,
        # better sample.
        if not outcome.passed:
            return outcome

        accepted, dup_meta = self.deduper.accept(source_image, edited_image)
        outcome.metrics.update({k: round(v, 4) for k, v in dup_meta.items()})
        if not accepted:
            outcome.reject("image.duplicate")

        return outcome

    def check_text_pair(self, source_caption: str, edited_caption: str) -> FilterOutcome:
        """Lexical sanity check used when BERTScore is not installed."""
        outcome = FilterOutcome(passed=True)
        outcome.metrics["token_f1"] = round(token_f1(source_caption, edited_caption), 4)
        if outcome.metrics["token_f1"] < 0.5:
            outcome.reject("text.lexical_divergence")
        return outcome


# ------------------------------------------------------------------- selection
def quality_score(metrics: Dict[str, Any], target_ssim: float = 0.8923) -> float:
    """Rank surviving pairs when a task has more candidates than quota.

    Weighted so that instruction alignment dominates, structural preservation
    comes second, and locality of the change breaks ties. ``target_ssim`` is the
    mean reported in Section IV-B: pairs near it behave like the dataset the
    model was actually trained on.
    """
    clip_d = float(metrics.get("clip_d", 0.0))
    ssim_value = float(metrics.get("ssim", 0.0))
    share = float(metrics.get("largest_component_share", 0.0))
    structure = 1.0 - min(1.0, abs(ssim_value - target_ssim))
    return 0.5 * min(1.0, clip_d / 0.25) + 0.3 * structure + 0.2 * share


def select_for_quota(records: List[Dict[str, Any]], quota: int) -> List[Dict[str, Any]]:
    """Keep the ``quota`` best records and mark the rest as surplus."""
    ordered = sorted(records, key=lambda r: quality_score(r.get("filters", {}).get("metrics", {})), reverse=True)
    chosen, surplus = ordered[:quota], ordered[quota:]
    for record in chosen:
        record["status"] = "selected"
    for record in surplus:
        record["status"] = "surplus"
    return chosen


def summarise(outcomes: Iterable[FilterOutcome]) -> Dict[str, Any]:
    """Count reasons, so a run can be diagnosed without reading every line."""
    total = 0
    passed = 0
    reasons: Dict[str, int] = {}
    for outcome in outcomes:
        total += 1
        if outcome.passed:
            passed += 1
        for reason in outcome.violations:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total": total,
        "passed": passed,
        "rejected": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
