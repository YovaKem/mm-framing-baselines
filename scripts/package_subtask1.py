"""Package a split's articles as Subtask 1 input, per the shared-task data
formatting spec (docs/Multimodal_Framing-2.pdf, section 4.1): one `<doc-id>.txt`
(UTF-8, title as first line, paragraphs delimited by blank lines) per document,
using the same naming convention as Subtask 2 (scripts/package_subtask2.py):
`<lang>_<M/D>_<TRAIN/TEST>_<unique-id>`.

Usage: python3 package_subtask1.py --split {test,train_lora}  (default: test)

Also writes ground_truth_text_frames.tsv, mirroring section 2.1.4's own ground-truth
shape: `<file-name> <start-char-position> <end-char-position> <framing-dimension>*`.
There are no real per-paragraph span annotations in this project (frames are
document-level only, not per-paragraph), so the two position columns are dummy "None"
placeholders — and, same as Subtask 2's oracle_text_frames.tsv, multi-label rows are
randomly (seeded) split across two lines to emulate a document's frames being spread
across several paragraph-level annotation lines. Source is
`consolidated_text_generic_frame` (3-LLM-ensemble consensus): this project's own
text-generic-frame is silver-standard only, since no human-annotated text-frame label
exists at all (see package_subtask2.py's oracle docstring for the same caveat).

This IS the actual scoring target for Subtask 1 (unlike Subtask 2's
oracle_text_frames.tsv, which reuses the same labels only as optional 2b context) — so,
like ground_truth_image_frames.tsv, keep it on the organizer side, not in the
participant package.
"""
import argparse
import random

from common import DATA_DIR, CONSOLIDATED_PATH, RANDOM_SEED, read_jsonl
from package_subtask2 import SPLITS, CANONICAL_LABELS, format_article, frame_rows


def package_dir(split):
    return DATA_DIR / f"subtask1_{split.replace('_lora', '')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="test")
    args = parser.parse_args()

    naming_tag, expected_count = SPLITS[args.split]
    out_dir = package_dir(args.split)
    ground_truth_path = out_dir / "ground_truth_text_frames.tsv"

    rows = [r for r in read_jsonl(CONSOLIDATED_PATH) if r["split"] == args.split]
    assert len(rows) == expected_count, f"expected {expected_count} {args.split} rows, got {len(rows)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    ground_truth_lines = []

    for row in rows:
        doc_id = f"EN_M_{naming_tag}_{row['uuid']}"

        (out_dir / f"{doc_id}.txt").write_text(
            format_article(row["title"], row["article_text"]), encoding="utf-8"
        )

        labels = [CANONICAL_LABELS[k] for k in row["consolidated_text_generic_frame"]]
        ground_truth_lines.extend(frame_rows(doc_id, labels, rng))

    ground_truth_path.write_text("\n".join(ground_truth_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} .txt files to {out_dir}.")
    print(f"Wrote {len(ground_truth_lines)} rows to {ground_truth_path}")


if __name__ == "__main__":
    main()
