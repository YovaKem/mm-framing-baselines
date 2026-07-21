"""
Load copenlu/mm-framing and draw a reproducible random sample of 300 rows.

Defaults to the `valid_framing_subset` split (154k rows) — the paper's own
filtered, framing-analysis-ready subset (excludes "None"/invalid frame
predictions, articles <100 words, and Sports/Media topics). Pass --split full
to sample from the unfiltered 479k-row set instead.

Usage:
    python scripts/sample_dataset.py
    python scripts/sample_dataset.py --split full --n 300 --seed 42
"""
import argparse
import json

from datasets import load_dataset

from common import (
    HF_DATASET,
    HF_SPLIT,
    RANDOM_SEED,
    SAMPLE_PATH,
    SAMPLE_SIZE,
    parse_list_field,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=HF_SPLIT, choices=["full", "valid_framing_subset"])
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print(f"Loading {HF_DATASET} split={args.split} ...")
    ds = load_dataset(HF_DATASET, split=args.split)
    print(f"Loaded {len(ds):,} rows. Sampling {args.n} with seed={args.seed}.")

    shuffled = ds.shuffle(seed=args.seed)
    sample = shuffled.select(range(args.n))

    rows = []
    for row in sample:
        text_tags, text_err = parse_list_field(row.get("text-generic-frame"))
        img_tags, img_err = parse_list_field(row.get("img-generic-frame"))
        row["_text_generic_frame_parsed"] = text_tags
        row["_text_generic_frame_parse_error"] = text_err
        row["_img_generic_frame_parsed"] = img_tags
        row["_img_generic_frame_parse_error"] = img_err
        rows.append(row)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} sampled rows to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
