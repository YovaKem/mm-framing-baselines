"""
Compare our fresh relabeling (scripts/relabel_frames.py) against the
dataset's original text-generic-frame / img-generic-frame labels, and report
how much they agree.

Also reports, using ONLY the new labels, how often the image's frame set is a
subset of the text's frame set — the paper's/our own expectation, checked
against what the model actually produced rather than assumed.

Run after relabel_frames.py.

Usage:
    python scripts/report_overlap.py
"""
from collections import Counter

from common import (
    CANONICAL_FRAMES,
    OVERLAP_REPORT_PATH,
    RELABELED_PATH,
    normalize_frame_tags,
    parse_list_field,
    read_jsonl,
)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def label_name(key):
    return CANONICAL_FRAMES[key].split(" — ")[0]


def compare(old_raw, new_keys):
    old_keys, _ = normalize_frame_tags(parse_list_field(old_raw)[0])
    old_set, new_set = set(old_keys), set(new_keys)
    return {
        "old": old_set,
        "new": new_set,
        "jaccard": jaccard(old_set, new_set),
        "identical": old_set == new_set,
        "retained": old_set & new_set,
        "dropped": old_set - new_set,
        "added": new_set - old_set,
    }


def main():
    if not RELABELED_PATH.exists():
        raise SystemExit(f"{RELABELED_PATH} not found — run scripts/relabel_frames.py first")
    rows = read_jsonl(RELABELED_PATH)
    n = len(rows)

    text_comparisons, img_comparisons = [], []
    dropped_text, added_text = Counter(), Counter()
    dropped_img, added_img = Counter(), Counter()
    img_subset_count, img_nonempty_count, img_none_new, img_none_old = 0, 0, 0, 0

    for row in rows:
        tc = compare(row.get("text-generic-frame"), row.get("new_text_generic_frame") or [])
        text_comparisons.append(tc)
        dropped_text.update(tc["dropped"])
        added_text.update(tc["added"])

        ic = compare(row.get("img-generic-frame"), row.get("new_img_generic_frame") or [])
        img_comparisons.append(ic)
        dropped_img.update(ic["dropped"])
        added_img.update(ic["added"])

        new_img = set(row.get("new_img_generic_frame") or [])
        new_text = set(row.get("new_text_generic_frame") or [])
        if not new_img:
            img_none_new += 1
        else:
            img_nonempty_count += 1
            if new_img <= new_text:
                img_subset_count += 1
        if not (row.get("img-generic-frame") or "").strip() or set(parse_list_field(row.get("img-generic-frame"))[0]) == set():
            img_none_old += 1

    def summarize(comparisons, label):
        avg_jaccard = sum(c["jaccard"] for c in comparisons) / len(comparisons)
        identical = sum(c["identical"] for c in comparisons)
        completely_different = sum(1 for c in comparisons if c["jaccard"] == 0 and (c["old"] or c["new"]))
        avg_old_size = sum(len(c["old"]) for c in comparisons) / len(comparisons)
        avg_new_size = sum(len(c["new"]) for c in comparisons) / len(comparisons)
        lines = [
            f"### {label}",
            f"- Average Jaccard overlap (old vs new): **{avg_jaccard:.2f}**",
            f"- Identical label sets: {identical}/{n} ({identical / n:.0%})",
            f"- Completely disjoint (no shared labels, at least one non-empty): {completely_different}/{n} ({completely_different / n:.0%})",
            f"- Average labels per row — old: {avg_old_size:.1f}, new: {avg_new_size:.1f}",
        ]
        return "\n".join(lines)

    lines = [f"# Relabeling overlap report\n\nRows compared: {n}\n"]
    lines.append(summarize(text_comparisons, "text-generic-frame"))
    lines.append("")
    lines.append(summarize(img_comparisons, "img-generic-frame"))

    lines.append("\n### Most frequently dropped from text (present in old, absent in new)")
    for key, count in dropped_text.most_common(10):
        lines.append(f"- {label_name(key)}: {count}")
    lines.append("\n### Most frequently added to text (absent in old, present in new)")
    for key, count in added_text.most_common(10):
        lines.append(f"- {label_name(key)}: {count}")

    lines.append("\n### Most frequently dropped from image (present in old, absent in new)")
    for key, count in dropped_img.most_common(10):
        lines.append(f"- {label_name(key)}: {count}")
    lines.append("\n### Most frequently added to image (absent in old, present in new)")
    for key, count in added_img.most_common(10):
        lines.append(f"- {label_name(key)}: {count}")

    lines.append("\n### Image-is-subset-of-text check (NEW labels only)")
    lines.append(f"- New image frame set is empty (\"None\"): {img_none_new}/{n} ({img_none_new / n:.0%}) "
                 f"— vs. {img_none_old}/{n} ({img_none_old / n:.0%}) under the old labels")
    if img_nonempty_count:
        lines.append(f"- Of the {img_nonempty_count} rows with a non-empty new image frame set, "
                     f"{img_subset_count} ({img_subset_count / img_nonempty_count:.0%}) are a subset of the new text frame set — "
                     f"the remaining {img_nonempty_count - img_subset_count} have the image conveying a frame the text doesn't.")

    report = "\n".join(lines)
    OVERLAP_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nWrote full report to {OVERLAP_REPORT_PATH}")


if __name__ == "__main__":
    main()
