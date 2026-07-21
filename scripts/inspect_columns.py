"""
Inspect the sampled rows: column-by-column non-null counts, cardinality, and a
frequency breakdown of the text/image generic-frame taxonomy actually observed
in the sample (cross-checked against the 15-category taxonomy from the paper).

Run after sample_dataset.py (and, optionally, scrape_articles.py).

Usage:
    python scripts/inspect_columns.py
"""
from collections import Counter

import pandas as pd

from common import (
    CANONICAL_FRAMES,
    INSPECTION_REPORT_PATH,
    SAMPLE_PATH,
    SCRAPED_PATH,
    normalize_frame_tags,
    read_jsonl,
)

COLUMN_DESCRIPTIONS = {
    "uuid": "Unique ID for the article.",
    "title": "Article headline.",
    "date_publish": "Publication timestamp.",
    "source_domain": "Publisher's domain (28 unique outlets in the full dataset).",
    "url": "Original article URL — the only pointer back to the source text/image; not itself the content.",
    "political_leaning": "Publisher bias label: left / left_lean / center / right_lean / right.",
    "text-topic": "Main subject of the article (LLM-derived, short phrase).",
    "text-topic-exp": "LLM's free-text justification for the topic label.",
    "text-entity-name": "Key entity/person/org the article centers on (LLM-derived).",
    "text-entity-sentiment": "Sentiment expressed toward that entity in the text (LLM-derived).",
    "text-entity-sentiment-exp": "LLM's free-text justification for the entity sentiment.",
    "text-generic-frame": "MAIN LABEL — multi-label set of generic frames present in the article text, drawn from the 15-category Boydstun et al. taxonomy. Stored as a stringified Python list of short tags.",
    "text-generic-frame-exp": "LLM's free-text justification for the text frame labels.",
    "text-issue-frame": "Issue-specific (non-fixed-vocabulary) framing label for the text, e.g. 'Geopolitical Tension'.",
    "text-issue-frame-exp": "LLM's free-text justification for the issue frame label.",
    "img-generic-frame": "MAIN LABEL — multi-label set of generic frames present in the article's lead image, same 15-category taxonomy as text-generic-frame. Stringified Python list of short tags.",
    "img-frame-exp": "LLM's free-text justification for the image frame labels.",
    "img-entity-name": "Key entity/subject visible in the image (LLM-derived).",
    "img-entity-sentiment": "Sentiment conveyed by the image toward that entity (LLM-derived).",
    "img-entity-sentiment-exp": "LLM's free-text justification for the image entity sentiment.",
    "gpt-topic": "Separate GPT-generated topic classification (broader taxonomy than text-topic).",
}


def load_sample_df():
    path = SCRAPED_PATH if SCRAPED_PATH.exists() else SAMPLE_PATH
    if not path.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found — run scripts/sample_dataset.py first")
    rows = read_jsonl(path)
    return pd.DataFrame(rows), path


def frame_tag_report(df, column):
    all_known, all_unknown = Counter(), Counter()
    rows_with_unknown = 0
    for tags in df[f"_{column.replace('-', '_')}_parsed"]:
        canonical, unknown = normalize_frame_tags(tags)
        all_known.update(canonical)
        if unknown:
            rows_with_unknown += 1
            all_unknown.update(unknown)
    return all_known, all_unknown, rows_with_unknown


def main():
    df, path = load_sample_df()
    lines = [f"# Inspection report\n\nSource: `{path.name}`  |  Rows: {len(df)}\n"]

    lines.append("## Columns\n")
    lines.append("| column | non-null | unique | description |")
    lines.append("|---|---|---|---|")
    for col in df.columns:
        if col.startswith("_") or col in {
            "scrape_http_status", "scrape_error", "article_text", "article_word_count",
            "image_local_path", "image_error",
        }:
            continue
        non_null = df[col].notna().sum()
        nunique = df[col].nunique(dropna=True)
        desc = COLUMN_DESCRIPTIONS.get(col, "")
        lines.append(f"| `{col}` | {non_null}/{len(df)} | {nunique} | {desc} |")

    lines.append("\n## political_leaning distribution\n")
    lines.append(df["political_leaning"].value_counts(dropna=False).to_string())

    lines.append("\n\n## Top 15 gpt-topic values\n")
    lines.append(df["gpt-topic"].value_counts(dropna=False).head(15).to_string())

    for col in ["text-generic-frame", "img-generic-frame"]:
        known, unknown, rows_with_unknown = frame_tag_report(df, col)
        lines.append(f"\n\n## {col}: canonical tag frequency (of {len(CANONICAL_FRAMES)} taxonomy categories)\n")
        for key, count in known.most_common():
            lines.append(f"- {CANONICAL_FRAMES[key].split(' — ')[0]}: {count}")
        missing = set(CANONICAL_FRAMES) - set(known)
        if missing:
            lines.append(f"\nNever observed in this sample: {', '.join(sorted(missing))}")
        if unknown:
            lines.append(f"\n**{rows_with_unknown} row(s) contained tags NOT in the known taxonomy** (needs review):")
            for tag, count in unknown.most_common():
                lines.append(f"- {tag!r}: {count}")

    for col in ["_text_generic_frame_parse_error", "_img_generic_frame_parse_error"]:
        n_errors = df[col].notna().sum()
        if n_errors:
            lines.append(f"\n\n**{col}: {n_errors} row(s) failed to parse** — see flag_issues.py output.")

    report = "\n".join(str(l) for l in lines)
    INSPECTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSPECTION_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nWrote full report to {INSPECTION_REPORT_PATH}")


if __name__ == "__main__":
    main()
