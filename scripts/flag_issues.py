"""
Sweep over the consolidated rows (scripts/consolidate_annotations.py output) to
flag rows worth a closer manual look, so the human validator isn't reviewing
every row uniformly. Layers:

1. Structural/statistical checks on the original dataset columns (malformed
   list literals, out-of-taxonomy tags, suspiciously thin LLM explanations,
   out-of-range dates) — cheap, deterministic.
2. A per-model text-relabel call that errored (text is still ensemble-derived;
   image is the paper's own human double-annotation, not a per-model call).
3. Low agreement among the 3 ensemble models on TEXT frames (worth a look
   regardless of what the majority vote landed on), a consolidated image
   frame set that isn't a subset of the consolidated text frame set (useful
   signal, not necessarily wrong), and a large disagreement between the
   consolidated result and the dataset's original label.

Run after consolidate_annotations.py.

Usage:
    python scripts/flag_issues.py
"""
import json
from collections import Counter
from datetime import datetime

from common import (
    CONSOLIDATED_PATH,
    ENSEMBLE_MODELS,
    ENTITY_SENTIMENT_VALUES,
    FLAGS_PATH,
    POLITICAL_LEANING_VALUES,
    model_slug,
    normalize_frame_tags,
    parse_list_field,
    read_jsonl,
)

EXP_FIELDS = [
    "text-topic-exp",
    "text-entity-sentiment-exp",
    "text-generic-frame-exp",
    "text-issue-frame-exp",
    "img-frame-exp",
    "img-entity-sentiment-exp",
]
SHORT_EXP_THRESHOLD = 15  # characters
LOW_OVERLAP_JACCARD = 0.25  # below this, old vs new share little/nothing


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_row(row, seen_uuids, seen_titles):
    flags = []

    def flag(code, detail=""):
        flags.append({"code": code, "detail": detail})

    uuid = row.get("uuid")
    if not uuid:
        flag("missing_uuid")
    elif uuid in seen_uuids:
        flag("duplicate_uuid")
    else:
        seen_uuids.add(uuid)

    title = (row.get("title") or "").strip()
    if not title:
        flag("missing_title")
    elif title in seen_titles:
        flag("duplicate_title", title)
    else:
        seen_titles.add(title)

    if not (row.get("url") or "").strip():
        flag("missing_url")

    # --- original generic frame fields: structural sanity only ---
    for prefix in ["text", "img"]:
        tags, parse_err = parse_list_field(row.get(f"{prefix}-generic-frame"))
        if parse_err:
            flag(f"original_{prefix}_generic_frame_parse_error", parse_err)
        else:
            _, unknown = normalize_frame_tags(tags)
            if unknown:
                flag(f"original_{prefix}_generic_frame_unknown_tag", ", ".join(sorted(set(unknown))))

    # --- entity sentiment / metadata sanity on the original columns ---
    for field in ["text-entity-sentiment", "img-entity-sentiment"]:
        val = (row.get(field) or "").strip()
        if val and val.lower() not in ENTITY_SENTIMENT_VALUES:
            flag(f"{field.replace('-', '_')}_unrecognized_value", val)

    leaning = (row.get("political_leaning") or "").strip()
    if leaning and leaning.lower() not in POLITICAL_LEANING_VALUES:
        flag("political_leaning_unrecognized_value", leaning)

    date_str = (row.get("date_publish") or "").strip()
    if date_str:
        parsed_date = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        if parsed_date is None:
            flag("date_publish_unparseable", date_str)
        elif parsed_date.year not in (2023, 2024):
            flag("date_publish_out_of_expected_range", date_str)
    else:
        flag("missing_date_publish")

    for field in EXP_FIELDS:
        val = (row.get(field) or "").strip()
        if val and len(val) < SHORT_EXP_THRESHOLD:
            flag("suspiciously_short_explanation", f"{field}={val!r}")

    # --- per-model relabel call outcomes (text only — image is human ground truth,
    # not ensemble-relabeled, so there's no per-model image call to have failed) ---
    by_model = row.get("by_model") or {}
    for model in ENSEMBLE_MODELS:
        slug = model_slug(model)
        info = by_model.get(slug, {})
        if info.get("text_exp") is None:
            flag("relabel_text_call_failed", slug)

    # --- ensemble agreement (text only) ---
    text_sets = [set(m["text_frames"]) for m in by_model.values()]
    text_pairwise = [jaccard(text_sets[i], text_sets[j]) for i in range(len(text_sets)) for j in range(i + 1, len(text_sets))]
    if text_pairwise and sum(text_pairwise) / len(text_pairwise) < LOW_OVERLAP_JACCARD:
        flag("low_ensemble_agreement_text", f"avg pairwise jaccard={sum(text_pairwise) / len(text_pairwise):.2f}")

    consolidated_text = set(row.get("consolidated_text_generic_frame") or [])
    consolidated_img = set(row.get("consolidated_img_generic_frame") or [])
    if consolidated_img and not consolidated_img <= consolidated_text:
        extra = ", ".join(sorted(consolidated_img - consolidated_text))
        flag("image_frame_not_subset_of_text", f"image conveys frame(s) not in the text: {extra}")

    # Compares against the dataset's OWN original label (Pixtral/Mistral zero-shot,
    # per docs/DATASET.md) — a QA signal, not a correctness check: for the image side
    # this is now real human ground truth vs. the paper's own auto-label, which is a
    # more interesting divergence than it used to be.
    old_text_keys, _ = normalize_frame_tags(parse_list_field(row.get("text-generic-frame"))[0])
    old_img_keys, _ = normalize_frame_tags(parse_list_field(row.get("img-generic-frame"))[0])
    text_jaccard = jaccard(old_text_keys, consolidated_text)
    img_jaccard = jaccard(old_img_keys, consolidated_img)
    if text_jaccard < LOW_OVERLAP_JACCARD:
        flag("text_frame_diverges_from_original", f"jaccard={text_jaccard:.2f}")
    if img_jaccard < LOW_OVERLAP_JACCARD:
        flag("img_frame_diverges_from_original", f"jaccard={img_jaccard:.2f}")

    return flags


def main():
    if not CONSOLIDATED_PATH.exists():
        raise SystemExit(f"{CONSOLIDATED_PATH} not found — run scripts/consolidate_annotations.py first")
    rows = read_jsonl(CONSOLIDATED_PATH)

    seen_uuids, seen_titles = set(), set()
    flags_by_uuid = {}
    code_counter = Counter()
    flagged_row_count = 0

    for row in rows:
        row_flags = check_row(row, seen_uuids, seen_titles)
        flags_by_uuid[row["uuid"]] = row_flags
        if row_flags:
            flagged_row_count += 1
            code_counter.update(f["code"] for f in row_flags)

    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(flags_by_uuid, f, indent=2, ensure_ascii=False)

    print(f"Source: {CONSOLIDATED_PATH.name}")
    print(f"Flagged {flagged_row_count}/{len(rows)} rows ({flagged_row_count / len(rows):.0%})\n")
    print("Flag code frequency:")
    for code, count in code_counter.most_common():
        print(f"  {count:4d}  {code}")
    print(f"\nWrote per-row flags to {FLAGS_PATH}")


if __name__ == "__main__":
    main()
