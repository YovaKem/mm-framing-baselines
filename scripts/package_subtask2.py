"""Package a split's articles/images as Subtask 2 input, per the shared-task data
formatting spec (docs/Multimodal_Framing-2.pdf, sections 4.1/4.2): one `<doc-id>.txt`
(UTF-8, title as first line, paragraphs delimited by blank lines) and one `<doc-id>.jpg`
image per document, sharing a doc id built from the naming convention:
`<lang>_<M/D>_<TRAIN/TEST>_<unique-id>`.

Usage: python3 package_subtask2.py --split {test,train_lora}  (default: test)

Article text is the real scraped `article_text` (available for every row in both
splits) — no placeholder needed there. What genuinely doesn't exist in this dataset is
a human/gold-standard TEXT-frame label (only `human_img_generic_frame` is
human-annotated; the text side is silver-standard only, via
`consolidated_text_generic_frame`'s 3-LLM-ensemble consensus) — relevant for anyone
building the Subtask 2b (oracle) condition on top of this package, since the spec's own
"oracle" is defined as human-labeled gold-standard.

Images outside the spec's size bounds (shortest side >= 512px, longest side <= 2048px)
are resized to fit before packaging; everything else is copied byte-for-byte.

Also writes two TSVs, each with each frame in its own tab-separated column (matching
section 2.1.4's own convention: "framing dimension labels are separated by tabs as
well"). A document with no frame gets no extra columns, matching this project's
"None" = empty list convention.

- oracle_text_frames.tsv — the 2b-only oracle context, keyed to `<doc-id>.txt`. Source
  is `consolidated_text_generic_frame` (3-LLM-ensemble consensus) — the closest thing to
  an oracle this dataset has, since no human-annotated text-frame label exists at all.
  Mirrors section 2.1.4's own ground-truth shape,
  `<file-name> <start-char-position> <end-char-position> <framing-dimension>*`, since the
  oracle *is* Subtask 1's ground truth — but we have no real per-paragraph span offsets,
  so those two columns are dummy "None" placeholders. To emulate that a real annotation
  file can carry a document's frames across multiple paragraph-level lines rather than
  one line per document, multi-label rows are randomly (seeded) split across two lines.
- ground_truth_image_frames.tsv — the actual scoring target, keyed to `<doc-id>.jpg`, one
  row per document. Source is `human_img_generic_frame`, the paper's own human
  double-annotation labels. NOT for participant release — keep on the organizer side.
"""
import argparse
import io
import random
import re

from PIL import Image

from common import DATA_DIR, CONSOLIDATED_PATH, RANDOM_SEED, read_jsonl, CANONICAL_FRAMES

# split value (in sample_consolidated.jsonl) -> (naming-convention TRAIN/TEST tag, expected row count)
SPLITS = {
    "test": ("TEST", 300),
    "train_lora": ("TRAIN", 100),
}

SPLIT_ROW_PROBABILITY = 0.5  # chance a multi-label oracle row is split across two lines

CANONICAL_LABELS = {k: v.split(" — ")[0] for k, v in CANONICAL_FRAMES.items()}

MIN_SHORT_SIDE = 512
MAX_LONG_SIDE = 2048


def fit_image_bytes(src_path):
    """Return (jpeg_bytes, was_resized) satisfying the spec's size bounds."""
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        short, long_ = min(w, h), max(w, h)
        scale = 1.0
        if long_ > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / long_
        elif short < MIN_SHORT_SIDE:
            scale = MIN_SHORT_SIDE / short
        resized = scale != 1.0
        if resized:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), resized


def format_article(title, article_text):
    """Title as the first line, then paragraphs delimited by a blank line each,
    per section 4.1's format. The scraped text uses single newlines between
    paragraphs, so those are normalized to blank-line separators here."""
    paragraphs = re.sub(r"\n+", "\n\n", article_text.strip())
    return f"{title}\n\n{paragraphs}\n"


def frame_rows(doc_id, labels, rng):
    """One or two tab-separated rows for a document's text-frame labels: `<file-name>
    None None <framing-dimension>*`, split across two lines if there are >= 2 labels
    and the random draw says so. Used both for Subtask 2's oracle_text_frames.tsv and
    (scripts/package_subtask1.py) Subtask 1's ground_truth_text_frames.tsv — same
    underlying labels, same section 2.1.4 shape, two different downstream purposes."""
    if len(labels) >= 2 and rng.random() < SPLIT_ROW_PROBABILITY:
        split_at = rng.randint(1, len(labels) - 1)
        parts = [labels[:split_at], labels[split_at:]]
    else:
        parts = [labels]
    return ["\t".join([f"{doc_id}.txt", "None", "None"] + part) for part in parts]


def doc_id_to_uuid(doc_id):
    # doc_id is "EN_M_<TRAIN|TEST>_<uuid>"; the uuid itself never contains underscores,
    # so splitting on the first 3 gives it cleanly.
    return doc_id.split("_", 3)[3]


def parse_article_txt(path):
    """Inverse of format_article above: first line is the title, the rest (after
    the blank line) is the article body."""
    text = path.read_text(encoding="utf-8")
    title, _, body = text.partition("\n\n")
    return title.strip(), body.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="test")
    args = parser.parse_args()

    naming_tag, expected_count = SPLITS[args.split]
    out_dir = DATA_DIR / f"subtask2_{args.split.replace('_lora', '')}"
    oracle_path = out_dir / "oracle_text_frames.tsv"
    ground_truth_path = out_dir / "ground_truth_image_frames.tsv"

    rows = [r for r in read_jsonl(CONSOLIDATED_PATH) if r["split"] == args.split]
    assert len(rows) == expected_count, f"expected {expected_count} {args.split} rows, got {len(rows)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    n_resized = 0
    oracle_lines = []
    ground_truth_lines = []

    for row in rows:
        doc_id = f"EN_M_{naming_tag}_{row['uuid']}"
        src_image_path = DATA_DIR / row["image_local_path"]

        (out_dir / f"{doc_id}.txt").write_text(
            format_article(row["title"], row["article_text"]), encoding="utf-8"
        )

        jpeg_bytes, resized = fit_image_bytes(src_image_path)
        (out_dir / f"{doc_id}.jpg").write_bytes(jpeg_bytes)
        n_resized += resized

        oracle_labels = [CANONICAL_LABELS[k] for k in row["consolidated_text_generic_frame"]]
        oracle_lines.extend(frame_rows(doc_id, oracle_labels, rng))

        gt_cols = [f"{doc_id}.jpg"] + [CANONICAL_LABELS[k] for k in row["human_img_generic_frame"]]
        ground_truth_lines.append("\t".join(gt_cols))

    oracle_path.write_text("\n".join(oracle_lines) + "\n", encoding="utf-8")
    ground_truth_path.write_text("\n".join(ground_truth_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} .txt/.jpg pairs to {out_dir} ({n_resized} images resized).")
    print(f"Wrote {len(oracle_lines)} rows to {oracle_path}")
    print(f"Wrote {len(ground_truth_lines)} rows to {ground_truth_path}")


if __name__ == "__main__":
    main()
