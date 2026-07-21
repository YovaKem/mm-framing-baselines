"""
Consolidate the 3-model ensemble (common.ENSEMBLE_MODELS) into a single label
set per row: a frame is kept only if at least AGREEMENT_THRESHOLD (2 of 3) of
the models independently assigned it, for text and image separately. Each
individual model's raw output is kept alongside for transparency.

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
        raise SystemExit(f"{NEWS_SAMPLE_PATH} not found — run scripts/filter_news.py first")

    per_model = {}
    for model in ENSEMBLE_MODELS:
        path = relabel_model_path(model)
        if not path.exists():
            raise SystemExit(f"{path} not found — run: python scripts/relabel_frames.py --model {model}")
        per_model[model] = {r["uuid"]: r for r in read_jsonl(path)}

    base_rows = read_jsonl(NEWS_SAMPLE_PATH)
    consolidated_rows = []
    text_pairwise_jaccards, img_pairwise_jaccards = [], []
    text_unanimous, img_unanimous = 0, 0
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
                "text_frame_strengths": res.get("new_text_generic_frame_strengths") or {},
                "text_exp": res.get("new_text_generic_frame_exp"),
                "img_frames": res.get("new_img_generic_frame") or [],
                "img_frame_strengths": res.get("new_img_generic_frame_strengths") or {},
                "img_exp": res.get("new_img_generic_frame_exp"),
            }
            if res.get("new_text_generic_frame_error"):
                call_errors[f"{slug}_text"] += 1
            if res.get("new_img_generic_frame_error"):
                call_errors[f"{slug}_img"] += 1

        text_sets = [set(m["text_frames"]) for m in by_model.values()]
        img_sets = [set(m["img_frames"]) for m in by_model.values()]

        consolidated_text, text_votes = consolidate_field(
            {mdl: per_model[mdl].get(uuid, {}) for mdl in ENSEMBLE_MODELS}, "new_text_generic_frame"
        )
        consolidated_img, img_votes = consolidate_field(
            {mdl: per_model[mdl].get(uuid, {}) for mdl in ENSEMBLE_MODELS}, "new_img_generic_frame"
        )

        out_row = dict(row)
        out_row["by_model"] = by_model
        out_row["consolidated_text_generic_frame"] = consolidated_text
        out_row["consolidated_text_frame_votes"] = text_votes
        out_row["consolidated_img_generic_frame"] = consolidated_img
        out_row["consolidated_img_frame_votes"] = img_votes
        consolidated_rows.append(out_row)

        text_kept_counts.append(len(consolidated_text))
        img_kept_counts.append(len(consolidated_img))
        if all(s == text_sets[0] for s in text_sets[1:]):
            text_unanimous += 1
        if all(s == img_sets[0] for s in img_sets[1:]):
            img_unanimous += 1
        for i in range(len(text_sets)):
            for j in range(i + 1, len(text_sets)):
                text_pairwise_jaccards.append(jaccard(text_sets[i], text_sets[j]))
                img_pairwise_jaccards.append(jaccard(img_sets[i], img_sets[j]))

    write_jsonl(CONSOLIDATED_PATH, consolidated_rows)

    n = len(consolidated_rows)
    lines = [f"# Consolidation report\n\nModels: {', '.join(ENSEMBLE_MODELS)}",
             f"Agreement threshold: {AGREEMENT_THRESHOLD} of {len(ENSEMBLE_MODELS)}",
             f"Rows: {n}\n"]
    lines.append(f"### Text frames")
    lines.append(f"- Avg pairwise Jaccard across model pairs: {sum(text_pairwise_jaccards) / len(text_pairwise_jaccards):.2f}")
    lines.append(f"- Rows where all {len(ENSEMBLE_MODELS)} models produced identical sets: {text_unanimous}/{n} ({text_unanimous / n:.0%})")
    lines.append(f"- Avg frames kept per row after {AGREEMENT_THRESHOLD}-of-{len(ENSEMBLE_MODELS)} consolidation: {sum(text_kept_counts) / n:.2f}")
    lines.append(f"- Rows with zero frames surviving consolidation: {sum(1 for c in text_kept_counts if c == 0)}/{n}")

    lines.append(f"\n### Image frames")
    lines.append(f"- Avg pairwise Jaccard across model pairs: {sum(img_pairwise_jaccards) / len(img_pairwise_jaccards):.2f}")
    lines.append(f"- Rows where all {len(ENSEMBLE_MODELS)} models produced identical sets: {img_unanimous}/{n} ({img_unanimous / n:.0%})")
    lines.append(f"- Avg frames kept per row after {AGREEMENT_THRESHOLD}-of-{len(ENSEMBLE_MODELS)} consolidation: {sum(img_kept_counts) / n:.2f}")
    lines.append(f"- Rows with zero frames surviving consolidation (\"None\"): {sum(1 for c in img_kept_counts if c == 0)}/{n}")

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
