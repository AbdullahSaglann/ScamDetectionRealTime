#!/usr/bin/env python3
"""Clean exact and near-duplicate rows from Sonhali_augmented.csv.

The script keeps the first occurrence of each normalized text, then removes
highly similar rows within the same label to reduce augmentation leakage and
overfitting risk.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/archive/Sonhali_augmented.csv")
OUTPUT_FILE = Path("data/intermediate/Sonhali_augmented_clean.csv")
REMOVED_FILE = Path("data/intermediate/Sonhali_augmented_removed_near_duplicates.csv")

SIMILARITY_THRESHOLD = 0.94
MAX_HAMMING_DISTANCE = 16


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


def word_shingles(words: list[str], size: int = 3) -> list[str]:
    if len(words) < size:
        return words
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def simhash(features: list[str], bits: int = 64) -> int:
    vector = [0] * bits
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        hashed = int.from_bytes(digest, "big")
        for bit in range(bits):
            vector[bit] += 1 if (hashed >> bit) & 1 else -1

    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_candidates(texts: list[str], hashes: list[int]) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for row_index, hashed in enumerate(hashes):
        length_bucket = len(texts[row_index]) // 80
        for band in range(8):
            band_value = (hashed >> (band * 8)) & 0xFF
            buckets[(band, band_value, length_bucket)].append(row_index)

    candidates: set[tuple[int, int]] = set()
    for bucket_rows in buckets.values():
        if len(bucket_rows) < 2 or len(bucket_rows) >= 1000:
            continue

        for left_pos, left in enumerate(bucket_rows):
            for right in bucket_rows[left_pos + 1 :]:
                max_len = max(len(texts[left]), len(texts[right]))
                if abs(len(texts[left]) - len(texts[right])) <= max(80, 0.25 * max_len):
                    candidates.add((left, right) if left < right else (right, left))

    return candidates


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    original_rows = len(df)
    df["_normalized_text"] = df["text"].map(normalize_text)

    exact_clean = (
        df.drop_duplicates(subset=["_normalized_text"], keep="first")
        .reset_index()
        .rename(columns={"index": "original_index"})
    )
    exact_removed = original_rows - len(exact_clean)

    texts = exact_clean["_normalized_text"].tolist()
    labels = exact_clean["label"].tolist()
    feature_sets = []
    for text in texts:
        shingles = word_shingles(tokenize(text), size=3)
        feature_sets.append(shingles if shingles else [text])
    hashes = [simhash(features) for features in feature_sets]

    candidates = build_candidates(texts, hashes)
    remove_rows: dict[int, tuple[int, float]] = {}

    for left, right in sorted(candidates):
        if labels[left] != labels[right]:
            continue
        if hamming_distance(hashes[left], hashes[right]) > MAX_HAMMING_DISTANCE:
            continue

        similarity = SequenceMatcher(None, texts[left], texts[right], autojunk=False).ratio()
        if similarity >= SIMILARITY_THRESHOLD and right not in remove_rows:
            remove_rows[right] = (left, similarity)

    keep_mask = [row_index not in remove_rows for row_index in range(len(exact_clean))]
    clean = exact_clean.loc[keep_mask, ["text", "label"]].reset_index(drop=True)

    removed_records = []
    for removed_index, (kept_index, similarity) in remove_rows.items():
        removed_records.append(
            {
                "removed_original_index": exact_clean.loc[removed_index, "original_index"],
                "kept_original_index": exact_clean.loc[kept_index, "original_index"],
                "label": exact_clean.loc[removed_index, "label"],
                "similarity": round(similarity, 4),
                "removed_text": exact_clean.loc[removed_index, "text"],
                "kept_text": exact_clean.loc[kept_index, "text"],
            }
        )

    clean.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(removed_records).to_csv(REMOVED_FILE, index=False)

    print("Input rows:", original_rows)
    print("After exact duplicate cleanup:", len(exact_clean))
    print("Exact duplicates removed:", exact_removed)
    print("Near duplicates removed:", len(remove_rows))
    print("Clean rows:", len(clean))
    print()
    print("Old label counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print()
    print("Clean label counts:")
    print(clean["label"].value_counts().sort_index().to_string())
    print()
    print("Clean label distribution:")
    print(clean["label"].value_counts(normalize=True).sort_index().to_string())
    print()
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Wrote: {REMOVED_FILE}")


if __name__ == "__main__":
    main()
