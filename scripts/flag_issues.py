"""
Sweep over the 300 sampled rows to flag likely issues before manual review, so
the human validator is pointed at what's worth their time instead of all 300
uniformly. Two layers:

1. Structural/statistical checks (missing fields, malformed list literals,
   out-of-taxonomy tags, suspiciously thin LLM explanations, out-of-range
   dates) — cheap, deterministic, no model calls.
2. Semantic verdicts from scripts/judge_frames.py (data/frame_judgments.json),
   if present — an LLM's independent judgment of whether text-generic-frame /
   img-generic-frame actually holds up against the real article text/image,
   merged in here as flags with the judge's reasoning as the detail.

Run after build_sample.py and (for the semantic layer) judge_frames.py.

Usage:
    python scripts/flag_issues.py
"""
import json
from collections import Counter
from datetime import datetime

from common import (
    ENTITY_SENTIMENT_VALUES,
    FLAGS_PATH,
    FRAME_JUDGMENTS_PATH,
    POLITICAL_LEANING_VALUES,
    SAMPLE_PATH,
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


def check_row(row, seen_uuids, seen_titles, judgment):
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
    for prefix in ["text", "img"]:
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

    # --- semantic verdicts from judge_frames.py, if available ---
    if judgment:
        if judgment.get("error"):
            flag("frame_judgment_failed", judgment["error"])
        else:
            if judgment.get("text_frame_verdict") == "questionable":
                flag("text_frame_questionable", judgment.get("text_frame_reasoning", ""))
            if judgment.get("img_frame_verdict") == "questionable":
                flag("img_frame_questionable", judgment.get("img_frame_reasoning", ""))

    return flags


def main():
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found — run scripts/build_sample.py first")
    rows = read_jsonl(SAMPLE_PATH)

    judgments = {}
    if FRAME_JUDGMENTS_PATH.exists():
        judgments = json.loads(FRAME_JUDGMENTS_PATH.read_text(encoding="utf-8"))
    else:
        print(f"NOTE: {FRAME_JUDGMENTS_PATH} not found — run scripts/judge_frames.py for the "
              "semantic accuracy check. Continuing with structural checks only.\n")

    seen_uuids, seen_titles = set(), set()
    flags_by_uuid = {}
    code_counter = Counter()
    flagged_row_count = 0

    for row in rows:
        row_flags = check_row(row, seen_uuids, seen_titles, judgments.get(row["uuid"]))
        flags_by_uuid[row["uuid"]] = row_flags
        if row_flags:
            flagged_row_count += 1
            code_counter.update(f["code"] for f in row_flags)

    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(flags_by_uuid, f, indent=2, ensure_ascii=False)

    print(f"Source: {SAMPLE_PATH.name}")
    print(f"Flagged {flagged_row_count}/{len(rows)} rows ({flagged_row_count / len(rows):.0%})\n")
    print("Flag code frequency:")
    for code, count in code_counter.most_common():
        print(f"  {count:4d}  {code}")
    print(f"\nWrote per-row flags to {FLAGS_PATH}")


if __name__ == "__main__":
    main()
