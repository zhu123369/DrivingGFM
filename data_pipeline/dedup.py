# -*- coding: utf-8 -*-
"""Duplicate control for captions and image pairs.

Text near-duplication is caught with 5-gram shingles and a MinHash signature;
image near-duplication with a 64-bit difference hash and a Hamming threshold.
Both are deterministic and dependency-free, so the same corpus always gives the
same keep/drop decision.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .metrics import dhash, hamming_distance


# ------------------------------------------------------------------- text side
def normalize(text: str) -> str:
    import re

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def shingles(text: str, width: int = 5) -> Set[str]:
    tokens = normalize(text).split()
    if len(tokens) < width:
        return {hashlib.sha1(" ".join(tokens).encode()).hexdigest()} if tokens else set()
    return {
        hashlib.sha1(" ".join(tokens[i : i + width]).encode()).hexdigest()
        for i in range(len(tokens) - width + 1)
    }


def jaccard(first: Set[str], second: Set[str]) -> float:
    if not first or not second:
        return 0.0
    intersection = len(first & second)
    union = len(first | second)
    return intersection / union if union else 0.0


class TextDeduper:
    """Reject a triple whose captions and instruction are already covered.

    ``threshold`` is a shingle Jaccard: 0.9 means "more than nine tenths of the
    five-grams are shared", i.e. a paraphrase of an existing pair rather than a
    new scene or a new edit.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = float(threshold)
        self.shingle_sets: List[Set[str]] = []
        self.exact: Set[str] = set()
        self.rejected = 0

    def accept(self, record: Dict[str, Any]) -> Tuple[bool, float]:
        signature = _record_signature(record)
        if signature in self.exact:
            self.rejected += 1
            return False, 1.0

        shingle = shingles(
            f"{record['source_caption']} {record['instruction']} {record['edited_caption']}"
        )
        best = 0.0
        for known in self.shingle_sets:
            score = jaccard(shingle, known)
            if score > best:
                best = score
                if best >= self.threshold:
                    break

        if best >= self.threshold:
            self.rejected += 1
            return False, best

        self.shingle_sets.append(shingle)
        self.exact.add(signature)
        return True, best


def _record_signature(record: Dict[str, Any]) -> str:
    payload = "|".join(
        normalize(record.get(field, ""))
        for field in ("source_caption", "instruction", "edited_caption")
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ image side
class ImageDeduper:
    """Track image hashes so a repeated frame is only counted once.

    ``max_edits_per_source`` allows a source frame to appear in more than one
    pair only when the edits genuinely differ; the default of 1 matches the
    dataset description, where each pair is an independent scene.
    """

    def __init__(self, hash_size: int = 8, max_hamming: int = 6, max_edits_per_source: int = 1):
        self.hash_size = int(hash_size)
        self.max_hamming = int(max_hamming)
        self.max_edits_per_source = int(max_edits_per_source)
        self.hashes: List[np.ndarray] = []
        self.source_counts: Dict[int, int] = {}
        self.rejected = 0

    def accept(self, source_image: Any, edited_image: Any) -> Tuple[bool, Dict[str, float]]:
        source_hash = dhash(source_image, self.hash_size)
        edited_hash = dhash(edited_image, self.hash_size)

        distances = [
            min(hamming_distance(source_hash, known), hamming_distance(edited_hash, known))
            for known in self.hashes
        ]
        nearest = min(distances) if distances else self.max_hamming + 1
        duplicate = nearest <= self.max_hamming

        source_id = int("".join("1" if bit else "0" for bit in source_hash.tolist()) or "0", 2)
        used = self.source_counts.get(source_id, 0)
        over_used = used >= self.max_edits_per_source

        if duplicate or over_used:
            self.rejected += 1
            return False, {
                "nearest_hamming": float(nearest),
                "duplicate": float(duplicate),
                "reused_source": float(over_used),
            }

        self.hashes.append(source_hash)
        self.source_counts[source_id] = used + 1
        return True, {"nearest_hamming": float(nearest), "duplicate": 0.0, "reused_source": 0.0}
