"""
Sweep over the relabeled rows (scripts/relabel_frames.py output) to flag rows
worth a closer manual look, so the human validator isn't reviewing all rows
uniformly. Two layers:

1. Structural/statistical checks on the original dataset columns (malformed
   list literals, out-of-taxonomy tags, suspiciously thin LLM explanations,
   out-of-range dates) — cheap, deterministic.
2. Signal from the relabeling itself: a relabel call that errored, an image
   frame set that isn't a subset of the text frame set (useful signal per
   the project's own expectation, not necessarily wrong), and a large
   old-vs-new disagreement (low Jaccard overlap) worth double-checking.

Run after relabel_frames.py.

Usage:
    python scripts/flag_issues.py
"""
import json
from collections import Counter
from datetime import datetime

from common import (
    ENTITY_SENTIMENT_VALUES,
    FLAGS_PATH,
    POLITICAL_LEANING_VALUES,
    RELABELED_PATH,
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

    # --- relabeling outcomes ---
    if row.get("new_text_generic_frame_error"):
        flag("relabel_text_call_failed", row["new_text_generic_frame_error"])
    if row.get("new_img_generic_frame_error"):
        flag("relabel_image_call_failed", row["new_img_generic_frame_error"])

    new_text = set(row.get("new_text_generic_frame") or [])
    new_img = set(row.get("new_img_generic_frame") or [])
    if new_img and not new_img <= new_text:
        extra = ", ".join(sorted(new_img - new_text))
        flag("image_frame_not_subset_of_text", f"image conveys frame(s) not in the text: {extra}")

    old_text_keys, _ = normalize_frame_tags(parse_list_field(row.get("text-generic-frame"))[0])
    old_img_keys, _ = normalize_frame_tags(parse_list_field(row.get("img-generic-frame"))[0])
    text_jaccard = jaccard(old_text_keys, new_text)
    img_jaccard = jaccard(old_img_keys, new_img)
    if text_jaccard < LOW_OVERLAP_JACCARD:
        flag("text_frame_diverges_from_original", f"jaccard={text_jaccard:.2f}")
    if img_jaccard < LOW_OVERLAP_JACCARD:
        flag("img_frame_diverges_from_original", f"jaccard={img_jaccard:.2f}")

    return flags


def main():
    if not RELABELED_PATH.exists():
        raise SystemExit(f"{RELABELED_PATH} not found — run scripts/relabel_frames.py first")
    rows = read_jsonl(RELABELED_PATH)

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

    print(f"Source: {RELABELED_PATH.name}")
    print(f"Flagged {flagged_row_count}/{len(rows)} rows ({flagged_row_count / len(rows):.0%})\n")
    print("Flag code frequency:")
    for code, count in code_counter.most_common():
        print(f"  {count:4d}  {code}")
    print(f"\nWrote per-row flags to {FLAGS_PATH}")


if __name__ == "__main__":
    main()
