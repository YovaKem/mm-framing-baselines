"""
Automated sweep over the 300 sampled rows to flag likely data-quality issues
BEFORE manual review, so the human validator can be pointed at the rows most
worth their time instead of all 300 uniformly.

Checks are intentionally structural/statistical (missing fields, malformed
list literals, out-of-taxonomy tags, suspiciously short LLM explanations,
scrape failures, out-of-range dates) — not semantic ("does this label look
wrong"), which is what the manual review step is for.

Run after sample_dataset.py and (ideally) scrape_articles.py.

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
    SAMPLE_PATH,
    SCRAPED_PATH,
    normalize_frame_tags,
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
MIN_ARTICLE_WORDS = 100  # matches the paper's own filtering threshold


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

    # --- generic frame fields (the main labels) ---
    for prefix, exp_field in [("text", "text-generic-frame-exp"), ("img", "img-generic-frame-exp")]:
        parse_err = row.get(f"_{prefix}_generic_frame_parse_error")
        tags = row.get(f"_{prefix}_generic_frame_parsed") or []
        if parse_err:
            flag(f"{prefix}_generic_frame_parse_error", parse_err)
        elif not tags:
            flag(f"{prefix}_generic_frame_empty")
        else:
            canonical, unknown = normalize_frame_tags(tags)
            if unknown:
                flag(f"{prefix}_generic_frame_unknown_tag", ", ".join(sorted(set(unknown))))

    if row.get("text-generic-frame") and not (row.get("text-issue-frame") or "").strip():
        flag("text_issue_frame_missing_but_generic_present")

    # --- entity sentiment validity ---
    for field in ["text-entity-sentiment", "img-entity-sentiment"]:
        val = (row.get(field) or "").strip()
        if val and val.lower() not in ENTITY_SENTIMENT_VALUES:
            flag(f"{field.replace('-', '_')}_unrecognized_value", val)

    # --- publisher metadata sanity ---
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

    # --- suspiciously thin LLM explanations ---
    for field in EXP_FIELDS:
        val = (row.get(field) or "").strip()
        if val and len(val) < SHORT_EXP_THRESHOLD:
            flag("suspiciously_short_explanation", f"{field}={val!r}")

    # --- scrape outcomes (only present if scrape_articles.py has been run) ---
    if "scrape_error" in row or "article_text" in row:
        word_count = row.get("article_word_count") or 0
        if not row.get("article_text"):
            flag("article_text_unavailable", row.get("scrape_error") or "unknown")
        elif word_count < MIN_ARTICLE_WORDS:
            flag("article_text_too_short", f"{word_count} words")

        if not row.get("image_local_path"):
            flag("image_unavailable", row.get("image_error") or "unknown")

    return flags


def main():
    path = SCRAPED_PATH if SCRAPED_PATH.exists() else SAMPLE_PATH
    if not path.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found — run scripts/sample_dataset.py first")
    rows = read_jsonl(path)

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

    print(f"Source: {path.name}")
    print(f"Flagged {flagged_row_count}/{len(rows)} rows ({flagged_row_count / len(rows):.0%})\n")
    print("Flag code frequency:")
    for code, count in code_counter.most_common():
        print(f"  {count:4d}  {code}")
    print(f"\nWrote per-row flags to {FLAGS_PATH}")


if __name__ == "__main__":
    main()
