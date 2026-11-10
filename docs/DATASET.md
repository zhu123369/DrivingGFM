# Dataset construction

This repository holds the code that produces the 80,000 paired training samples 
the checkpoint was instruction-tuned on, plus the rules that decide which samples 
are kept.

Nothing here ships data or weights. It ships the procedure, the prompts, the 
thresholds and the acceptance criteria, so the dataset can be regenerated or
audited sample by sample.

---

## 1. What the pipeline does

Six stages, each writing a file the next stage reads. Every stage is resumable
and every decision is recorded next to the sample it was made for. Stages 1–5
produce the rendered pairs; stage 6 stops after text filtering and emits the
prompts alone.

| Stage | Script | Input | Output | Needs |
|---|---|---|---|---|
| 1. Caption expansion | `scripts/01_expand_captions.py` | 40 seed captions + 10 task templates | `data/captions/raw/<task>.jsonl` | an OpenAI-compatible LLM endpoint |
| 2. Text filtering | `scripts/02_filter_captions.py` | raw triples | `data/captions/filtered/<task>.jsonl` | numpy, optional BERTScore |
| 3. Pair rendering | `scripts/03_generate_pairs.py` | text-passing triples | `data/pairs/<task>.jsonl` + two PNGs per pair | GPU, diffusers, SD3 weights |
| 4. Pair filtering | `scripts/04_filter_pairs.py` | rendered pairs | same ledger, `status` updated | optional CLIP on GPU |
| 5. Manifest + report | `scripts/05_report.py` | task ledgers | `data/manifest/*.jsonl`, `reports/dataset_quality.*` | numpy, PIL |
| 6. Prompt-only selection | `scripts/06_select_prompts.py` | skips rendering: selects exactly 80,000 text triples and writes `data/manifest/prompts_only.jsonl` | CPU |

Stage 1 and 2 run on a CPU-only machine; only stage 3 needs the GPU, and it is
the only stage that is expensive. That ordering is deliberate: a malformed caption
is rejected before it can cost a denoising pass.

Stage 6 is the alternative exit: it reads the stage-2 output, selects exactly the
quota per task (round-robin over subtasks, ranked by closeness to the reported
caption statistics) and writes `data/manifest/prompts_only.jsonl`. It never
touches `data/manifest/all.jsonl`, so a text-only selection can sit next to a
rendered one. Under `--require-full` it exits non-zero and lists the short tasks
rather than emitting a manifest that looks complete but is not.

One command runs stages 1–5:

```bash
python scripts/run_all.py --dry-run          # validate config, no model calls
python scripts/run_all.py                    # full run, in order
```

---

## 2. Per-task quotas

`configs/tasks.yaml` carries the ten tasks and their counts, taken from Table I of
the manuscript:

| task | family | pairs | edit mode | pair strategy |
|---|---|---|---|---|
| `detection_sign` | object detection | 11,000 | region | p2p |
| `detection_lane` | object detection | 7,000 | region | p2p |
| `detection_vehicle` | object detection | 9,000 | region | p2p |
| `detection_pedestrian` | object detection | 7,000 | region | p2p |
| `enhancement_rainy` | image enhancement | 5,000 | refine | p2p |
| `enhancement_snowy` | image enhancement | 6,000 | refine | p2p |
| `enhancement_foggy` | image enhancement | 4,000 | refine | p2p |
| `enhancement_lowlight` | image enhancement | 10,000 | refine | p2p |
| `super_resolution` | super-resolution | 8,000 | refine | p2p (degradation fallback) |
| `object_removal` | object removal | 13,000 | word_swap | p2p |
| **total** | | **80,000** | | |

`PipelineConfig.validate_quota()` fails loudly if the ten numbers stop summing to
80,000, so an edit to the table cannot silently change the dataset size.

**One inconsistency worth recording.** Section III-C of the manuscript describes
"9,000 pairs for rainy scenarios" in running text, while Table I lists 5,000.
Only the Table I column adds up to the stated 80,000 pairs
(34,000 + 25,000 + 8,000 + 13,000), so Table I is authoritative here and
`configs/tasks.yaml` follows it. If the manuscript text is corrected, change
`quota` for `enhancement_rainy` and the total in the same commit.

---

## 3. Stage 1 — caption, instruction, edited caption

### 3.1 Seeds

`prompts/seeds.yaml` holds 40 hand-written captions, one frame each. They are the
"several dozen input captions" of Section III-B. Each carries a `scene` block
(road type, time of day, weather, camera) and a `suitable_for` list that restricts
which task templates may expand it, so a foggy frame is never used to seed a
lane-marking edit that would contradict the caption.

Seeds are never written to the dataset verbatim; they steer the distribution.

### 3.2 Templates

`prompts/caption_expansion.yaml` contains:

* a system prompt fixing the role, seven hard rules and the JSON schema;
* one template per task, each supplying the allowed edit verbs, the class
  vocabulary and the variety axes to rotate through;
* two worked examples (the manuscript's speed-limit-sign case is one of them);
* `variety_axes` in two levels: scene axes shared by all tasks, and task axes
  (sign type, lane configuration, vehicle state, rain artefact, obstruction type,
  ...). These are what make 80,000 pairs diverse instead of 80,000 paraphrases;
* generation settings (temperature 0.9, top-p 0.95, 8 pairs per call).

The response contract is one JSON object:

```json
{"pairs": [{"task": "...", "subtask": "...", "source_caption": "...",
            "instruction": "...", "edited_caption": "...",
            "edited_region": "...", "edit_type": "word_swap|refine|region"}]}
```

`data_pipeline/prompts.py` strips reasoning-model scratchpads, recovers the JSON
from prose or fences, validates every triple, and drops the ones that do not
parse. A batch asking for 8 triples therefore returns fewer than 8 sometimes, and
stage 1 oversamples (`--oversample`, default 1.6) to compensate.

### 3.3 Endpoint

```bash
export DRIVINGGFM_LLM_BASE_URL=https://api.deepseek.com/v1
export DRIVINGGFM_LLM_API_KEY=...
export DRIVINGGFM_LLM_MODEL=deepseek-r1      # optional override
python scripts/01_expand_captions.py --workers 16
```

No vendor SDK is required: `data_pipeline/caption_expand.py` speaks the
chat-completions protocol over `urllib`, retries with backoff, and treats a non-429
4xx as a hard error rather than a transient one.

---

## 4. Stage 2 — text filtering rules

`configs/filter_rules.yaml` → `level_1_text`. Each rule exists because a specific
failure mode would otherwise reach the GPU and then the training set.

| Rule | Threshold | Why |
|---|---|---|
| caption length | 8–55 words, ≤3 sentences | one frame, not a trip |
| imperative opening | first word ∈ 23 edit verbs | a description is not an instruction |
| anchor in frame | one of 14 spatial patterns | grounds the edit; later drives the spatial mask |
| one edit only | no "and also", "then", "simultaneously" | a chained edit cannot be localised |
| scene preserved | ≤1 tracked scene noun may disappear | otherwise the caption pair is a re-generation |
| task consistency | condition/class term must appear in the source caption and the instruction | catches mislabelled pairs |
| safety deny-list | 11 patterns | see below |
| caption drift | BERTScore-F1 ∈ [0.88, 0.98] for source vs edited | 0.92 is the reported mean; 1.0 means nothing changed |
| instruction agreement | BERTScore-F1 ≥ 0.72 vs edited caption | 0.81 is the reported mean |
| duplicates | 5-gram MinHash Jaccard ≥ 0.90 | paraphrases inflate the count without adding scenes |

The safety deny-list is text-level pre-screening, not the "rule-constrained VLM
safety verification module" the manuscript lists as future work. It blocks the
cases a caption edit cannot legally or physically express:

* changing a traffic-light aspect (it defines the legal state of the scene);
* deleting pedestrians, cyclists or traffic signs, or "all vehicles";
* text overlays, watermarks and whole-frame regeneration.

### Fallback, stated rather than hidden

BERTScore needs a downloaded checkpoint. Without it the pipeline switches to a
lexical token-F1 and **uses different thresholds** (`fallback_when_unavailable`),
because 0.88 on a lexical scale would reject nearly everything. The substitution
is recorded in every sample's metadata as `metric: token_f1`, and the report
repeats the metric name. Nothing silently mixes the two scales.

---

## 5. Stage 3 — Prompt-to-Prompt on Stable Diffusion 3

### 5.1 Why this is not the published P2P implementation

Prompt-to-Prompt was written for latent diffusion UNets, where text and image
tokens meet in a cross-attention block that can be hooked and whose attention map
is an `(image tokens × text tokens)` matrix. Stable Diffusion 3 is a diffusion
transformer: text and latent patch tokens sit in **one** joint sequence, and
attention is computed over their concatenation. There is no cross-attention
processor to hook.

`data_pipeline/p2p.py` therefore implements the same idea in the terms the
architecture offers:

* the joint attention is recomputed explicitly, in query chunks, so the
  text-conditioning slice of the probability matrix is materialised and can be
  read and written;
* a source run records that slice for the first blocks over the early part of the
  schedule;
* the edited run replaces the edited slice with the recorded one, per row, for the
  same blocks and steps.

Algorithm 2 of the manuscript is what this implements: one shared noise sample,
two denoising trajectories, attention maps of the source merged into the edited
trajectory so the geometry survives and the change stays local.

### 5.2 Edit modes

| mode | used by | what is replaced |
|---|---|---|
| `refine` | four enhancement tasks, super-resolution | the whole text slice, inside the step window |
| `word_swap` | object removal, additions | only the columns of tokens that differ between the two prompts |
| `region` | four detection tasks | `word_swap` plus a spatial mask, so replacement is confined to the instructed area |

Word-level column selection is done by `TokenAligner`: it tokenises both prompts
with each of the three text tokenizers, aligns the token strings with `difflib`,
and maps edited positions onto source positions in the concatenated embedding
space. Inserted words borrow the attention of the token they were inserted after;
deleted words simply free their column.

If the source and edited prompts tokenise to different lengths — which happens
with an encoder set that pads to different maxima — the controller logs one
warning and stops replacing instead of writing a misaligned map. The pair is still
rendered and still has to pass the image filters, so a tokenisation mismatch
degrades the pair rather than corrupting the dataset.

### 5.3 Parameters

| flag | default | meaning |
|---|---|---|
| `--steps` | 25 | denoising steps per run |
| `--guidance` | 4.5 | SD3 classifier-free guidance |
| `--size` | 512 | square training resolution |
| `--replace-end` | 0.5 | attention is exchanged over the first half of the schedule |
| `--store-blocks` | `0,1,2` | joint blocks whose maps are recorded |
| `--strategy` | per task | `p2p` or `degradation` |
| `--dtype` | `float16` | weight dtype |

Memory: recorded maps live on the CPU in float16. One joint block at 512 px costs
`2 × heads × 4096 × 231 × 2 B ≈ 91 MB` per step with classifier-free guidance
(two rows in the batch; the controller stores both and maps row to row). Three
blocks over 13 of 25 steps is therefore about **3.5 GB of host RAM**. The exact
figure for a given run is written to each pair's `gen.stored_mb`, computed by
`AttentionController.store_bytes_mb()`. Raise `--store-blocks` only if host
memory allows it; on a 16 GB machine keep it at `0,1`.

### 5.4 Super-resolution fallback

`--strategy degradation` renders one frame and derives the source by a 4× bicubic
down/up-sample. That gives exact supervision for super-resolution without an
attention pass, which matters on a machine that cannot hold the recorded maps
alongside the transformer. It is a documented variant, not the default: the
default for all ten tasks follows the manuscript and uses Prompt-to-Prompt. The
per-pair `gen.strategy` field records which route produced the pair.

---

## 6. Stage 4 — image filtering rules

`level_2_image`, applied per pair:

| Rule | Threshold | Grounding |
|---|---|---|
| SSIM | ≥ 0.80 | reported mean is 0.8923; the floor keeps the tail of legitimately large edits |
| SSIM upper bound | ≤ 0.99 | a visually identical pair teaches nothing |
| changed area | 0.2%–35% of the frame | rejects a vanished edit and a rebuilt frame |
| largest component | ≥ 40% of changed pixels in one blob | rejects speckle spread over the frame |
| mean luma | 0.05–0.97, std ≥ 0.02 | rejects collapsed and blown-out outputs |
| channel shift | ≤ 0.35 mean per-channel | rejects colour cast from a failed attention exchange |
| CLIP-D | ≥ 0.08 | reported mean 0.2179, shuffled baseline 0.042 |
| duplicates | dHash Hamming ≤ 6; one edit per source | a repeated frame inflates the count |

Then `select_for_quota` keeps exactly `quota` pairs per task, ranked by

```
0.5 · CLIP-D (capped at 0.25) + 0.3 · (1 − |SSIM − 0.8923|) + 0.2 · largest_component_share
```

so a task with more candidates than quota keeps the pairs whose alignment is
strong and whose structure matches the reported dataset mean. Surplus pairs stay
in the ledger with `status: surplus` and are never deleted.

---

## 7. Stage 5 — manifest and quality report

`data/manifest/all.jsonl` is the training-side contract: one line per kept pair,
with two captions, the instruction, both image paths, the filter metrics and the
sampling metadata. `data/manifest/summary.json` holds the per-task counts, and
`verify_quota_table()` compares them with Table I.

`scripts/05_report.py` writes `reports/dataset_quality.json` plus a Markdown
summary containing:

* per-task counts against the quotas;
* mean BERTScore-F1 for source vs edited, and instruction vs edited;
* mean SSIM (from the filter metrics, or recomputed from images);
* mean CLIP-D with the **shuffled-correspondence baseline** and the bootstrap 95%
  CI, matching the analyses added in the revision;
* FID against KITTI and Cityscapes when those reference folders are available
  (`--fid-kitti`, `--fid-cityscapes`, needs `torch-fidelity`).

Every field carries `computed: true/false`. A machine without CLIP produces a
report that says the alignment row was not computed, rather than a report with a
weaker metric in the same field.

What the report deliberately does **not** claim: that any metric "ensures" the
factual correctness of an edit. It reports distribution-level statistics, which is
exactly the limitation the revision states and the reviewer asked for.

---

## 8. Reproducibility notes

**Deterministic given the inputs.** Sampling seeds per pair, the caption
deduplication, every filter threshold, quota selection and the manifest are all
deterministic. Re-running stages 2–5 on the same ledgers reproduces the same
kept set and the same report.

**Not bit-reproducible.** Two things vary and are recorded rather than hidden:

* caption generation samples from the LLM at temperature 0.9; the same seed yields
  the same offsets but a different service or model version yields different text.
  The `gen` block on each record names the model, temperature and template.
* GPU denoising is not bit-exact across hardware and cuDNN versions. SSIM and
  CLIP-D are stable to a few 1e-3 but not identical.

**Cost.** With 1.6× oversampling the caption stage asks for about 128,000 triples;
at 8 usable triples per call that is roughly 20,000 calls for the full table.
Pair rendering is two denoising runs per pair, so the dataset is ~160,000 runs at
25 steps and 512 px. Time it on one task first (`--tasks detection_sign --limit 50`).

---

## 9. Known limitations of this release

1. **No human evaluation harness.** The 200-pair human study and its Spearman
   correlation (0.68) in the revision need annotators; the repository ships the
   sampling code and the annotation form, not the labels.
2. **Text-level safety screening only.** The deny-list is not the rule-constrained
   VLM verification module; the manuscript lists it as future work and this
   repository does not pretend otherwise.
3. **Super-resolution pairs** default to Prompt-to-Prompt with a sharpening
   instruction, and offer the degradation route as a fallback. Both are recorded
   per pair; neither is a substitute for the Cityscapes evaluation protocol, which
   downsamples real images as described in Section IV-E.
4. **CLIP-D thresholds** are calibrated against the reported mean (0.2179) and
   shuffled baseline (0.042). A different CLIP checkpoint shifts the scale; the
   filter records the model name in `configs/filter_rules.yaml`.
5. **Table I vs Section III-C** rainy count, as described in §2 above.

---

## 10. Licence and provenance

The generated pairs derive from images synthesised by Stable Diffusion 3
(`stabilityai/stable-diffusion-3-medium`, CreativeML Open RAIL-M), so the
Attachment A use restrictions propagate to the derived dataset. The pipeline
introduces no personal data: captions describe scenes, never people, plates or
places, and the prompts forbid brands and identifiers.
