"""
Compare the local small-model baselines against the consolidated (2-of-3
ensemble majority vote) ground truth:

  - Qwen3-4B-Instruct zero-shot text framing vs. consolidated_text_generic_frame
  - Qwen3-VL-4B-Instruct zero-shot image framing (no-oracle: image + title
    only) vs. consolidated_img_generic_frame
  - Qwen3-VL-4B-Instruct zero-shot image framing (with-oracle: image + title +
    the ground-truth text frame) vs. consolidated_img_generic_frame, and
    against the no-oracle run, to see whether telling the model the correct
    text framing changes/improves its image predictions or just gets copied.

Run after baseline_qwen_text.py, and baseline_qwen_vlm.py --no-oracle /
--oracle.

Usage:
    python scripts/report_baselines.py
"""
from collections import Counter

from common import (
    CANONICAL_FRAMES,
    CONSOLIDATED_PATH,
    DATA_DIR,
    read_jsonl,
    relabel_model_path,
)
from baseline_qwen_vlm import MODEL_ID as VLM_MODEL_ID, baseline_vlm_path
from baseline_qwen_text import MODEL_ID as TEXT_MODEL_ID

REPORT_PATH = DATA_DIR / "baselines_report.md"


def label_name(key):
    return CANONICAL_FRAMES[key].split(" — ")[0]


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare(rows, gold_field, pred_by_uuid, pred_field, label):
    jaccards, identical, disjoint = [], 0, 0
    gold_sizes, pred_sizes = [], []
    missed, extra = Counter(), Counter()
    for row in rows:
        gold = set(row.get(gold_field) or [])
        pred = set((pred_by_uuid.get(row["uuid"], {}) or {}).get(pred_field) or [])
        jaccards.append(jaccard(gold, pred))
        if gold == pred:
            identical += 1
        if jaccards[-1] == 0 and (gold or pred):
            disjoint += 1
        gold_sizes.append(len(gold))
        pred_sizes.append(len(pred))
        missed.update(gold - pred)
        extra.update(pred - gold)

    n = len(rows)
    lines = [
        f"### {label}",
        f"- Avg Jaccard vs. consolidated ground truth: **{sum(jaccards) / n:.2f}**",
        f"- Identical label sets: {identical}/{n} ({identical / n:.0%})",
        f"- Completely disjoint (no shared labels, at least one non-empty): {disjoint}/{n} ({disjoint / n:.0%})",
        f"- Avg labels per row — ground truth: {sum(gold_sizes) / n:.2f}, baseline: {sum(pred_sizes) / n:.2f}",
        "",
        "Most-missed frames (in ground truth, baseline didn't predict):",
    ]
    for key, count in missed.most_common(8):
        lines.append(f"  - {label_name(key)}: {count}")
    lines.append("Most-over-predicted frames (baseline predicted, not in ground truth):")
    for key, count in extra.most_common(8):
        lines.append(f"  - {label_name(key)}: {count}")
    return "\n".join(lines), jaccards


def main():
    if not CONSOLIDATED_PATH.exists():
        raise SystemExit(f"{CONSOLIDATED_PATH} not found — run scripts/consolidate_annotations.py first")
    rows = read_jsonl(CONSOLIDATED_PATH)
    n = len(rows)

    text_path = relabel_model_path(TEXT_MODEL_ID)
    no_oracle_path = baseline_vlm_path(False)
    oracle_path = baseline_vlm_path(True)
    for p in (text_path, no_oracle_path, oracle_path):
        if not p.exists():
            raise SystemExit(f"{p} not found — run the corresponding baseline script first")

    text_by_uuid = {r["uuid"]: r for r in read_jsonl(text_path)}
    no_oracle_by_uuid = {r["uuid"]: r for r in read_jsonl(no_oracle_path)}
    oracle_by_uuid = {r["uuid"]: r for r in read_jsonl(oracle_path)}

    lines = [f"# Baseline report\n\nRows: {n}\n",
             f"Text model: `{TEXT_MODEL_ID}` (zero-shot, local)",
             f"Image model: `{VLM_MODEL_ID}` (zero-shot, local, two settings)\n"]

    section, _ = compare(rows, "consolidated_text_generic_frame", text_by_uuid, "new_text_generic_frame",
                          f"Text: {TEXT_MODEL_ID}")
    lines.append(section)

    section, no_oracle_jaccards = compare(rows, "consolidated_img_generic_frame", no_oracle_by_uuid,
                                           "new_img_generic_frame", f"Image (no oracle): {VLM_MODEL_ID}")
    lines.append("\n" + section)

    section, oracle_jaccards = compare(rows, "consolidated_img_generic_frame", oracle_by_uuid,
                                        "new_img_generic_frame", f"Image (with oracle text frame): {VLM_MODEL_ID}")
    lines.append("\n" + section)

    # Does oracle help, or just get copied?
    improved = sum(1 for a, b in zip(no_oracle_jaccards, oracle_jaccards) if b > a)
    worsened = sum(1 for a, b in zip(no_oracle_jaccards, oracle_jaccards) if b < a)
    unchanged = n - improved - worsened

    copy_count, oracle_nonempty = 0, 0
    for row in rows:
        oracle_text = set(row.get("consolidated_text_generic_frame") or [])
        oracle_img_pred = set((oracle_by_uuid.get(row["uuid"], {}) or {}).get("new_img_generic_frame") or [])
        if oracle_text:
            oracle_nonempty += 1
            if oracle_img_pred == oracle_text:
                copy_count += 1

    lines.append("\n### Does the oracle text frame help image prediction, or just get copied?")
    lines.append(f"- Avg Jaccard vs. ground truth — no oracle: {sum(no_oracle_jaccards) / n:.2f}, with oracle: {sum(oracle_jaccards) / n:.2f}")
    lines.append(f"- Rows where oracle setting improved agreement: {improved}/{n} ({improved / n:.0%})")
    lines.append(f"- Rows where oracle setting worsened agreement: {worsened}/{n} ({worsened / n:.0%})")
    lines.append(f"- Rows unchanged: {unchanged}/{n} ({unchanged / n:.0%})")
    lines.append(f"- Of {oracle_nonempty} rows with a non-empty oracle text frame, the oracle-setting image "
                 f"prediction was an EXACT copy of the text frame in {copy_count} ({copy_count / oracle_nonempty:.0%}) — "
                 f"a high rate here would suggest the model leans on the given text label rather than looking at the image.")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
