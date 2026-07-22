"""
Consolidate two different ground-truth sources into one row per article:

- TEXT frames: the 3-model ensemble's (common.ENSEMBLE_MODELS) 2-of-3
  majority vote — no human text-frame annotation exists, so this silver
  standard is unchanged from the original method. Each model's raw output is
  kept alongside (`by_model`) for transparency.
- IMAGE frames: the paper's own human double-annotation pass
  (data/human_image_frame_labels.csv, joined in by scripts/build_human_sample.py
  as `human_img_generic_frame`) — real ground truth, not a vote. Not an
  ensemble product, so there's no per-model breakdown or agreement stat for
  it; consolidation just carries it through as `consolidated_img_generic_frame`.

Run after relabel_frames.py has been run once per model in ENSEMBLE_MODELS
(writing to common.relabel_model_path(model) each time).

Usage:
    python scripts/consolidate_annotations.py
"""
from collections import Counter

from common import (
    AGREEMENT_THRESHOLD,
    CANONICAL_FRAMES,
    CONSOLIDATED_PATH,
    CONSOLIDATION_REPORT_PATH,
    ENSEMBLE_MODELS,
    NEWS_SAMPLE_PATH,
    model_slug,
    read_jsonl,
    relabel_model_path,
    write_jsonl,
)


def label_name(key):
    return CANONICAL_FRAMES[key].split(" — ")[0]


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def consolidate_field(per_model_results, field):
    """Majority-vote across models for one of new_text_generic_frame / new_img_generic_frame."""
    votes = Counter()
    for res in per_model_results.values():
        for key in res.get(field) or []:
            votes[key] += 1
    kept = [key for key, count in votes.items() if count >= AGREEMENT_THRESHOLD]
    return kept, dict(votes)


def main():
    if not NEWS_SAMPLE_PATH.exists():
        raise SystemExit(f"{NEWS_SAMPLE_PATH} not found — run scripts/build_human_sample.py first")

    per_model = {}
    for model in ENSEMBLE_MODELS:
        path = relabel_model_path(model)
        if not path.exists():
            raise SystemExit(f"{path} not found — run: python scripts/relabel_frames.py --model {model}")
        per_model[model] = {r["uuid"]: r for r in read_jsonl(path)}

    base_rows = read_jsonl(NEWS_SAMPLE_PATH)
    consolidated_rows = []
    text_pairwise_jaccards = []
    text_unanimous = 0
    text_kept_counts, img_kept_counts = [], []
    call_errors = Counter()

    for row in base_rows:
        uuid = row["uuid"]
        by_model = {}
        for model in ENSEMBLE_MODELS:
            res = per_model[model].get(uuid, {})
            slug = model_slug(model)
            by_model[slug] = {
                "text_frames": res.get("new_text_generic_frame") or [],
                "text_exp": res.get("new_text_generic_frame_exp"),
            }
            if res.get("new_text_generic_frame_error"):
                call_errors[f"{slug}_text"] += 1

        text_sets = [set(m["text_frames"]) for m in by_model.values()]

        consolidated_text, text_votes = consolidate_field(
            {mdl: per_model[mdl].get(uuid, {}) for mdl in ENSEMBLE_MODELS}, "new_text_generic_frame"
        )
        # Image ground truth is the paper's own human double-annotation (joined in by
        # build_human_sample.py), not an ensemble product — carried through as-is.
        consolidated_img = row.get("human_img_generic_frame") or []

        out_row = dict(row)
        out_row["by_model"] = by_model
        out_row["consolidated_text_generic_frame"] = consolidated_text
        out_row["consolidated_text_frame_votes"] = text_votes
        out_row["consolidated_img_generic_frame"] = consolidated_img
        consolidated_rows.append(out_row)

        text_kept_counts.append(len(consolidated_text))
        img_kept_counts.append(len(consolidated_img))
        if all(s == text_sets[0] for s in text_sets[1:]):
            text_unanimous += 1
        for i in range(len(text_sets)):
            for j in range(i + 1, len(text_sets)):
                text_pairwise_jaccards.append(jaccard(text_sets[i], text_sets[j]))

    write_jsonl(CONSOLIDATED_PATH, consolidated_rows)

    n = len(consolidated_rows)
    n_test = sum(1 for r in consolidated_rows if r.get("split") == "test")
    n_train = sum(1 for r in consolidated_rows if r.get("split") == "train_lora")
    lines = [f"# Consolidation report\n\nText ensemble models: {', '.join(ENSEMBLE_MODELS)}",
             f"Agreement threshold: {AGREEMENT_THRESHOLD} of {len(ENSEMBLE_MODELS)}",
             f"Rows: {n} (split=test: {n_test}, split=train_lora: {n_train})\n"]
    lines.append(f"### Text frames (silver standard: {AGREEMENT_THRESHOLD}-of-{len(ENSEMBLE_MODELS)} LLM ensemble consensus)")
    lines.append(f"- Avg pairwise Jaccard across model pairs: {sum(text_pairwise_jaccards) / len(text_pairwise_jaccards):.2f}")
    lines.append(f"- Rows where all {len(ENSEMBLE_MODELS)} models produced identical sets: {text_unanimous}/{n} ({text_unanimous / n:.0%})")
    lines.append(f"- Avg frames kept per row after {AGREEMENT_THRESHOLD}-of-{len(ENSEMBLE_MODELS)} consolidation: {sum(text_kept_counts) / n:.2f}")
    lines.append(f"- Rows with zero frames surviving consolidation: {sum(1 for c in text_kept_counts if c == 0)}/{n}")

    lines.append(f"\n### Image frames (ground truth: paper's own human double-annotation)")
    lines.append(f"- Avg frames per row: {sum(img_kept_counts) / n:.2f}")
    lines.append(f"- Rows with zero frames (\"None\"): {sum(1 for c in img_kept_counts if c == 0)}/{n} "
                 f"({sum(1 for c in img_kept_counts if c == 0) / n:.0%})")
    img_frame_freq = Counter()
    for row in consolidated_rows:
        img_frame_freq.update(row["consolidated_img_generic_frame"])
    lines.append("- Frame frequency: " + ", ".join(
        f"{label_name(k)} {c}" for k, c in img_frame_freq.most_common()
    ))

    if call_errors:
        lines.append(f"\n### Model call errors")
        for key, count in call_errors.most_common():
            lines.append(f"- {key}: {count}")

    report = "\n".join(lines)
    CONSOLIDATION_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nWrote {CONSOLIDATED_PATH} and {CONSOLIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
