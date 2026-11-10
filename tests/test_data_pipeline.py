# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data_pipeline import config as cfg
from data_pipeline.dedup import ImageDeduper, TextDeduper, jaccard, shingles
from data_pipeline.filtering import ImageFilter, TextFilter, quality_score, select_for_quota
from data_pipeline.manifest import build_manifest
from data_pipeline.metrics import (
    BertScorer,
    dhash,
    edit_region_stats,
    hamming_distance,
    ssim,
    token_f1,
)
from data_pipeline.prompts import build_messages, extract_json_object, parse_pairs, render_axes


SIGN_PAIR = {
    "task": "detection_sign",
    "subtask": "speed limit sign",
    "source_caption": (
        "The driver is travelling in the right-hand lane of a two-lane road; a "
        "circular speed limit sign reading 80 is mounted on a post on the right "
        "shoulder, about 40 m ahead, with a hedgerow behind it."
    ),
    "instruction": "Mark the speed limit sign on the right in blue.",
    "edited_caption": (
        "The driver is travelling in the right-hand lane of a two-lane road; a "
        "circular speed limit sign reading 80 is mounted on a post on the right "
        "shoulder, about 40 m ahead, now marked in blue, with a hedgerow behind it."
    ),
    "edited_region": "right shoulder, upper right of the frame",
    "edit_type": "region",
}

REMOVAL_PAIR = {
    "task": "object_removal",
    "subtask": "temporary barrier",
    "source_caption": (
        "A lane closure on an urban road; orange cones funnel traffic to the left, "
        "a temporary barrier blocks the right lane, and a works vehicle is parked "
        "behind the barrier."
    ),
    "instruction": "Remove the temporary barrier on the right side.",
    "edited_caption": (
        "A lane closure on an urban road; orange cones funnel traffic to the left, "
        "the right lane is now clear of the barrier, and a works vehicle is parked "
        "behind the cones."
    ),
    "edited_region": "right lane, lower right of the frame",
    "edit_type": "word_swap",
}


class ConfigTests(unittest.TestCase):
    def test_quota_matches_the_paper(self):
        tasks = cfg.load_tasks()
        tasks.validate_quota()
        self.assertEqual(tasks.total_quota(), 80000)
        self.assertEqual(tasks.task("detection_sign").quota, 11000)
        self.assertEqual(tasks.task("detection_lane").quota, 7000)
        self.assertEqual(tasks.task("detection_vehicle").quota, 9000)
        self.assertEqual(tasks.task("detection_pedestrian").quota, 7000)
        self.assertEqual(tasks.task("enhancement_rainy").quota, 5000)
        self.assertEqual(tasks.task("enhancement_snowy").quota, 6000)
        self.assertEqual(tasks.task("enhancement_foggy").quota, 4000)
        self.assertEqual(tasks.task("enhancement_lowlight").quota, 10000)
        self.assertEqual(tasks.task("super_resolution").quota, 8000)
        self.assertEqual(tasks.task("object_removal").quota, 13000)

    def test_quota_drift_is_rejected(self):
        raw = cfg._load_yaml(cfg.TASKS_CONFIG)
        raw["tasks"][0]["quota"] = 12000
        with self.assertRaises(ValueError):
            cfg.PipelineConfig(raw).validate_quota()

    def test_every_task_has_a_prompt_template_and_a_seed(self):
        tasks = cfg.load_tasks()
        prompts = cfg.load_caption_prompts()
        seeds = cfg.load_seeds()
        for spec in tasks.tasks:
            self.assertIn(spec.prompt_template, prompts["task_templates"])
            self.assertTrue(
                [s for s in seeds if spec.prompt_template in s.get("suitable_for", [])],
                f"{spec.id} has no seed caption",
            )


class PromptTests(unittest.TestCase):
    def test_templates_render_for_every_task(self):
        prompts = cfg.load_caption_prompts()
        for template_id in prompts["task_templates"]:
            messages = build_messages(prompts, template_id, 8, "1. a seed", "- an axis", 3)
            self.assertEqual(len(messages), 2)
            self.assertTrue(messages[0]["content"])
            self.assertIn("Output schema", messages[1]["content"])
            self.assertNotIn("{count}", messages[1]["content"])

    def test_unknown_template_is_refused(self):
        with self.assertRaises(KeyError):
            build_messages(cfg.load_caption_prompts(), "nope", 1, "", "", 0)

    def test_render_axes_handles_dicts(self):
        axes = [{"road type": "rural"}, "time of day: dusk"]
        text = render_axes(axes, __import__("random").Random(0), take=5)
        self.assertIn("road type", text)
        self.assertIn("dusk", text)

    def test_parse_fenced_json_with_reasoning_block(self):
        reply = (
            "<think>the sign is a good target</think>\n```json\n"
            '{"pairs": [' + json.dumps(SIGN_PAIR) + "]}\n```"
        )
        pairs = parse_pairs(reply, "detection_sign")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["task"], "detection_sign")

    def test_parse_drops_malformed_pairs(self):
        payload = {"pairs": [SIGN_PAIR, {"task": "detection_sign", "instruction": "x"}]}
        pairs = parse_pairs(json.dumps(payload), "detection_sign")
        self.assertEqual(len(pairs), 1)

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            extract_json_object("sorry, I cannot help with that")

    def test_unknown_edit_type_is_dropped(self):
        bad = dict(SIGN_PAIR, edit_type="recolor")
        payload = {"pairs": [bad]}
        pairs = parse_pairs(json.dumps(payload), "detection_sign")
        self.assertEqual(pairs, [])


class TextFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = cfg.load_filter_rules()
        cls.tasks = cfg.load_tasks()
        cls.filter = TextFilter(
            cls.rules, cls.tasks, bert=BertScorer(enabled=False), deduper=TextDeduper()
        )

    def test_good_pair_passes(self):
        outcome = self.filter.check(dict(SIGN_PAIR))
        self.assertTrue(outcome.passed, outcome.violations)
        self.assertIn("bertscore_source_edited", outcome.metrics)

    def test_removal_pair_passes_without_condition_in_source(self):
        outcome = TextFilter(
            self.rules, self.tasks, bert=BertScorer(enabled=False), deduper=TextDeduper()
        ).check(dict(REMOVAL_PAIR))
        self.assertTrue(outcome.passed, outcome.violations)

    def test_missing_anchor_is_rejected(self):
        record = dict(SIGN_PAIR, instruction="Mark it in blue.")
        outcome = self.filter.check(record)
        self.assertFalse(outcome.passed)
        self.assertIn("text.no_anchor", outcome.violations)

    def test_chained_edit_is_rejected(self):
        record = dict(
            SIGN_PAIR,
            instruction="Mark the speed limit sign on the right in blue and also remove the hedge.",
        )
        outcome = self.filter.check(record)
        self.assertFalse(outcome.passed)
        self.assertTrue(any(v.startswith("text.multi_edit") for v in outcome.violations))

    def test_removing_a_stop_sign_is_denied(self):
        record = dict(
            REMOVAL_PAIR,
            instruction="Remove the stop sign on the right side.",
            edited_caption="A lane closure on an urban road; the right lane is clear and the works vehicle is parked behind the cones.",
        )
        outcome = self.filter.check(record)
        self.assertFalse(outcome.passed)
        self.assertTrue(any(v.startswith("safety.denied") for v in outcome.violations))

    def test_traffic_light_aspect_is_denied(self):
        record = dict(
            SIGN_PAIR,
            instruction="Make the traffic light green on the right side of the road.",
        )
        outcome = self.filter.check(record)
        self.assertFalse(outcome.passed)

    def test_condition_mismatch_is_rejected(self):
        record = dict(
            SIGN_PAIR,
            task="enhancement_rainy",
            source_caption="A three-lane urban road at night; a silver sedan is 20 m ahead in the middle lane, its tail lights sharp against the asphalt, and the shop fronts on the right are clearly visible.",
            instruction="Remove the rain streaks on the right side and make the scene clear.",
            edited_caption="A three-lane urban road at night; a silver sedan is 20 m ahead in the middle lane, its tail lights sharp against the asphalt, and the shop fronts on the right are visible after the rain streaks are gone.",
        )
        outcome = self.filter.check(record)
        self.assertFalse(outcome.passed)
        self.assertIn("text.condition_absent_from_caption", outcome.violations)

    def test_not_imperative_is_rejected(self):
        record = dict(SIGN_PAIR, instruction="The sign on the right is blue.")
        outcome = self.filter.check(record)
        self.assertFalse(outcome.passed)
        self.assertIn("text.not_imperative", outcome.violations)

    def test_duplicate_is_rejected_after_first(self):
        filter_engine = TextFilter(
            self.rules, self.tasks, bert=BertScorer(enabled=False), deduper=TextDeduper()
        )
        first = filter_engine.check(dict(SIGN_PAIR))
        second = filter_engine.check(dict(SIGN_PAIR))
        self.assertTrue(first.passed)
        self.assertFalse(second.passed)
        self.assertIn("text.duplicate", second.violations)


class MetricTests(unittest.TestCase):
    def _frame(self):
        rng = np.random.default_rng(0)
        base = np.zeros((128, 128, 3), dtype=np.float32)
        base[..., 0] = 0.4
        base[..., 1] = 0.45
        base[..., 2] = 0.5
        base += rng.normal(0, 0.02, base.shape).astype(np.float32)
        return np.clip(base, 0, 1)

    def test_ssim_identity(self):
        frame = self._frame()
        self.assertAlmostEqual(ssim(frame, frame.copy()), 1.0, places=4)

    def test_ssim_drops_with_structural_change(self):
        frame = self._frame()
        changed = frame.copy()
        changed[20:60, 20:60] = 0.9
        self.assertLess(ssim(frame, changed), 0.95)

    def test_edit_region_locality(self):
        frame = self._frame()
        changed = frame.copy()
        changed[40:52, 40:52, 0] = 0.95
        stats = edit_region_stats(frame, changed)
        self.assertGreater(stats["area_fraction"], 0.0)
        self.assertLess(stats["area_fraction"], 0.05)
        self.assertGreater(stats["largest_component_share"], 0.5)
        self.assertIsNotNone(stats["bbox"])

    def test_scattered_edit_is_reported_as_scattered(self):
        frame = self._frame()
        changed = frame.copy()
        changed[::4, ::4, 0] = 0.95
        stats = edit_region_stats(frame, changed)
        self.assertLess(stats["largest_component_share"], 0.4)

    def test_dhash_near_duplicates(self):
        frame = self._frame()
        noisy = np.clip(frame + 0.01, 0, 1)
        self.assertLessEqual(hamming_distance(dhash(frame), dhash(noisy)), 6)

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("a b c", "a b c"), 1.0)
        self.assertLess(token_f1("a b c", "x y z"), 0.1)


class ImageFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = cfg.load_filter_rules()
        cls.engine = ImageFilter(cls.rules, clip=None, deduper=ImageDeduper())

    def _frames(self):
        rng = np.random.default_rng(1)
        base = np.full((96, 96, 3), 0.35, dtype=np.float32)
        base += rng.normal(0, 0.03, base.shape).astype(np.float32)
        base = np.clip(base, 0, 1)
        edited = base.copy()
        edited[20:40, 60:80, :] = 0.85
        return base, np.clip(edited, 0, 1)

    def test_local_edit_passes(self):
        source, edited = self._frames()
        outcome = self.engine.check(source, edited, "mark the sign on the right in blue", clip_d=0.15)
        self.assertTrue(outcome.passed, outcome.violations)

    def test_identical_images_are_rejected(self):
        source, _ = self._frames()
        outcome = self.engine.check(source, source.copy())
        self.assertFalse(outcome.passed)
        self.assertIn("image.no_change", outcome.violations)

    def test_whole_frame_change_is_rejected(self):
        source, _ = self._frames()
        outcome = self.engine.check(source, np.clip(source * 0.2, 0, 1))
        self.assertFalse(outcome.passed)

    def test_low_clip_d_is_rejected(self):
        source, edited = self._frames()
        outcome = self.engine.check(source, edited, "", clip_d=0.01)
        self.assertFalse(outcome.passed)
        self.assertIn("image.instruction_misaligned", outcome.violations)


class SelectionTests(unittest.TestCase):
    def test_quality_score_prefers_alignment(self):
        strong = {"clip_d": 0.22, "ssim": 0.89, "largest_component_share": 0.8}
        weak = {"clip_d": 0.09, "ssim": 0.89, "largest_component_share": 0.8}
        self.assertGreater(quality_score(strong), quality_score(weak))

    def test_select_for_quota_keeps_best_and_marks_surplus(self):
        records = [
            {"filters": {"metrics": {"clip_d": 0.10, "ssim": 0.9, "largest_component_share": 0.7}}},
            {"filters": {"metrics": {"clip_d": 0.20, "ssim": 0.9, "largest_component_share": 0.7}}},
            {"filters": {"metrics": {"clip_d": 0.15, "ssim": 0.9, "largest_component_share": 0.7}}},
        ]
        chosen = select_for_quota(records, 2)
        self.assertEqual([r["status"] for r in chosen], ["selected", "selected"])
        self.assertEqual(sum(1 for r in records if r["status"] == "surplus"), 1)


class DedupTests(unittest.TestCase):
    def test_shingles_and_jaccard(self):
        a = shingles("the car is parked on the left kerb near the shop")
        b = shingles("the car is parked on the left kerb near the shop")
        c = shingles("a freight train crosses the level crossing at dusk")
        self.assertAlmostEqual(jaccard(a, b), 1.0)
        self.assertLess(jaccard(a, c), 0.1)

    def test_text_deduper_rejects_paraphrase(self):
        deduper = TextDeduper(threshold=0.9)
        first = dict(SIGN_PAIR)
        second = dict(SIGN_PAIR, instruction="Mark the speed limit sign on the right in blue.")
        self.assertTrue(deduper.accept(first)[0])
        self.assertFalse(deduper.accept(second)[0])

    def test_image_deduper_rejects_same_frame_twice(self):
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        frame[:, :, 0] = np.linspace(0, 1, 64)[None, :]
        edited = frame.copy()
        edited[10:20, 10:20] = 1.0
        deduper = ImageDeduper()
        self.assertTrue(deduper.accept(frame, edited)[0])
        self.assertFalse(deduper.accept(frame, edited)[0])


class ManifestTests(unittest.TestCase):
    def test_manifest_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = cfg.ensure_dir(cfg.pairs_dir(root))
            tasks = cfg.load_tasks()
            spec = tasks.task("detection_sign")
            record = {
                "pair_id": "abc123",
                "task": spec.id,
                "subtask": "speed limit sign",
                "source_caption": SIGN_PAIR["source_caption"],
                "instruction": SIGN_PAIR["instruction"],
                "edited_caption": SIGN_PAIR["edited_caption"],
                "image_source": "data/pairs/detection_sign/abc123_src.png",
                "image_edited": "data/pairs/detection_sign/abc123_edit.png",
                "gen": {"strategy": "p2p"},
                "filters": {"image": {"passed": True, "metrics": {"ssim": 0.91, "clip_d": 0.21}}},
                "status": "selected",
            }
            cfg.append_jsonl(pairs / f"{spec.id}.jsonl", record)

            summary = build_manifest(root=root, tasks_cfg=tasks)
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["tasks"][spec.id]["count"], 1)
            self.assertFalse(summary["complete"])

            entries = cfg.read_jsonl(cfg.manifest_dir(root) / "all.jsonl")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["pair_id"], "abc123")


def _load_selector():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "06_select_prompts.py"
    spec = importlib.util.spec_from_file_location("select_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = _load_selector()

    def _records(self, subtasks, per_bucket):
        records = []
        for subtask in subtasks:
            for index in range(per_bucket):
                records.append({
                    "subtask": subtask,
                    "source_caption": f"caption about {subtask} number {index}",
                    "instruction": "mark it on the right",
                    "edited_caption": f"caption about {subtask} number {index} now marked",
                    "status": "text_pass",
                    "filters": {"text": {"metrics": {
                        "metric": "bertscore_f1",
                        "bertscore_source_edited": 0.92 - 0.01 * index,
                        "bertscore_instruction_edited": 0.81,
                    }}},
                })
        return records

    def test_selection_fills_quota_round_robin(self):
        records = self._records(["speed limit sign", "crosswalk"], per_bucket=5)
        chosen, stats = self.selector.select_task(records, 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(stats["subtasks"], 2)
        # Two per bucket, so neither subtask dominates the quota.
        self.assertEqual(len({r["subtask"] for r in chosen}), 2)
        self.assertEqual(sum(1 for r in chosen if r["subtask"] == "crosswalk"), 2)

    def test_selection_prefers_closeness_to_reported_mean(self):
        records = self._records(["speed limit sign"], per_bucket=3)
        chosen, _ = self.selector.select_task(records, 1)
        self.assertAlmostEqual(
            chosen[0]["filters"]["text"]["metrics"]["bertscore_source_edited"], 0.92
        )

    def test_selection_reports_shortfall(self):
        records = self._records(["speed limit sign"], per_bucket=2)
        chosen, stats = self.selector.select_task(records, 10)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(stats["selected"], 2)

    def test_fallback_metric_uses_its_own_targets(self):
        metrics = {
            "metric": "token_f1",
            "bertscore_source_edited": 0.93,
            "bertscore_instruction_edited": 0.35,
        }
        self.assertAlmostEqual(self.selector.text_quality_score(metrics), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
