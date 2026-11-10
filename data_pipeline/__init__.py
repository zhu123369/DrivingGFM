# -*- coding: utf-8 -*-
"""DrivingGFM data construction package.

Stage order (see ``docs/DATASET.md``):

    1. ``caption_expand``  seed captions -> (caption, instruction, caption) triples
    2. ``filtering``       text-level rules, so bad triples never reach the GPU
    3. ``pair_generation`` Prompt-to-Prompt sampling -> paired images
    4. ``filtering``       image-level rules
    5. ``report``          dataset quality statistics for Section IV-B

Heavy dependencies (torch, diffusers, transformers, CLIP, BERTScore) are imported
lazily inside the functions that need them, so the text-level stages and the
rule engine can be run and unit-tested on a machine with numpy only.
"""

__version__ = "1.0"
