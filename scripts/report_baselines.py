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

Metrics follow the original paper's own methodology (arXiv:2503.20960):
micro-averaged precision/recall/F1 (computed across all row x frame pairs,
matching their reported "micro averaged F1 score of 0.5" for text), plus a
non-zero-intersection rate (their "95.7%"/"84.2%" figures). Unlike the paper,
our image labels allow an explicit "None" (empty set) — the non-zero-
intersection rate is computed only over rows with a non-empty gold label
(matching the paper's original definition, since their data had no empty
labels), with a separate "None-agreement rate" reported for rows where the
gold label genuinely is empty.

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


def per_row_f1(gold, pred):
    """Per-row F1 with the standard 0/0 convention: both empty -> 1.0 (perfect
    agreement on "nothing applies"), only one empty or no overlap -> 0.0."""
    if not gold and not pred:
        return 1.0
    tp = len(gold & pred)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def compare(rows, gold_field, pred_by_uuid, pred_field):
    pairs = [
        (set(row.get(gold_field) or []), set((pred_by_uuid.get(row["uuid"], {}) or {}).get(pred_field) or []))
        for row in rows
    ]
    return compare_pairs(pairs)


def strong_gold_frames(row, modality):
    """Frames the CONSOLIDATED target holds at "strong" strength: kept only if
    >=2 of the 3 ensemble models rated that specific frame "strong" (not just
    "moderate") for this row, per modality ("text" or "img")."""
    by_model = row.get("by_model") or {}
    frame_field = f"{modality}_frames"
    strength_field = f"{modality}_frame_strengths"
    counts = Counter()
    for m in by_model.values():
        for f in m.get(frame_field, []):
            if m.get(strength_field, {}).get(f) == "strong":
                counts[f] += 1
    return {f for f, c in counts.items() if c >= 2}


def strong_pred_frames(pred_row, frame_field, strength_field):
    """The baseline's own frames it rated "strong" (drops its "moderate" ones)."""
    frames = pred_row.get(frame_field) or []
    strengths = pred_row.get(strength_field) or {}
    return {f for f in frames if strengths.get(f) == "strong"}


def compare_pairs(pairs):
    tp = fp = fn = 0
    identical = 0
    gold_sizes, pred_sizes = [], []
    missed, extra = Counter(), Counter()
    nzi_hits, nzi_total = 0, 0  # non-zero intersection, over rows with non-empty gold
    none_gold_total, none_gold_correct = 0, 0  # rows where gold IS empty ("None")
    f1s = []

    for gold, pred in pairs:
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        if gold == pred:
            identical += 1
        gold_sizes.append(len(gold))
        pred_sizes.append(len(pred))
        missed.update(gold - pred)
        extra.update(pred - gold)
        f1s.append(per_row_f1(gold, pred))

        if gold:
            nzi_total += 1
            if gold & pred:
                nzi_hits += 1
        else:
            none_gold_total += 1
            if not pred:
                none_gold_correct += 1

    n = len(pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "identical": identical,
        "nzi_hits": nzi_hits,
        "nzi_total": nzi_total,
        "none_gold_total": none_gold_total,
        "none_gold_correct": none_gold_correct,
        "avg_gold_size": sum(gold_sizes) / n,
        "avg_pred_size": sum(pred_sizes) / n,
        "missed": missed,
        "extra": extra,
        "per_row_f1": f1s,
    }


def summary_table(settings, n):
    lines = [
        "| Baseline | Precision | Recall | F1 | Non-zero intersection | Identical | Avg labels (gold) | Avg labels (pred) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in settings:
        nzi = f"{s['nzi_hits']}/{s['nzi_total']} ({s['nzi_hits']/s['nzi_total']:.0%})" if s["nzi_total"] else "n/a"
        lines.append(
            f"| {name} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {nzi} | "
            f"{s['identical']}/{n} ({s['identical']/n:.0%}) | {s['avg_gold_size']:.2f} | {s['avg_pred_size']:.2f} |"
        )
    return "\n".join(lines)


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

    text_stats = compare(rows, "consolidated_text_generic_frame", text_by_uuid, "new_text_generic_frame")
    no_oracle_stats = compare(rows, "consolidated_img_generic_frame", no_oracle_by_uuid, "new_img_generic_frame")
    oracle_stats = compare(rows, "consolidated_img_generic_frame", oracle_by_uuid, "new_img_generic_frame")

    settings = [
        (f"Text — {TEXT_MODEL_ID}", text_stats),
        (f"Image, no oracle — {VLM_MODEL_ID}", no_oracle_stats),
        (f"Image, with oracle — {VLM_MODEL_ID}", oracle_stats),
    ]

    lines = [
        "# Baseline report\n",
        f"Rows: {n}\n",
        "Metrics follow the original paper's methodology: micro-averaged precision/recall/F1 "
        "(computed across all row x frame pairs) and a non-zero-intersection rate. The paper's data "
        "never had an empty (\"None\") gold label, so non-zero-intersection is computed only over "
        "rows with a non-empty gold label here too; rows where gold genuinely is empty get their own "
        "\"None-agreement\" stat instead.\n",
    ]

    lines.append("## Summary\n")
    lines.append(summary_table(settings, n))

    none_rows = [(name, s) for name, s in settings if s["none_gold_total"]]
    if none_rows:
        lines.append("\n### \"None\" (empty gold label) agreement\n")
        lines.append("| Baseline | Rows where gold is None | Baseline also predicted None |")
        lines.append("|---|---:|---:|")
        for name, s in none_rows:
            lines.append(
                f"| {name} | {s['none_gold_total']}/{n} | "
                f"{s['none_gold_correct']}/{s['none_gold_total']} ({s['none_gold_correct']/s['none_gold_total']:.0%}) |"
            )

    # Strong-only: keep a gold frame only if >=2 of 3 ensemble models rated it "strong"
    # (not just "moderate"), and a baseline's own prediction only where IT rated a frame
    # "strong". Same three settings, filtered down to each side's high-confidence frames.
    text_strong_pairs = [
        (
            strong_gold_frames(row, "text"),
            strong_pred_frames(text_by_uuid.get(row["uuid"], {}), "new_text_generic_frame", "new_text_generic_frame_strengths"),
        )
        for row in rows
    ]
    no_oracle_strong_pairs = [
        (
            strong_gold_frames(row, "img"),
            strong_pred_frames(no_oracle_by_uuid.get(row["uuid"], {}), "new_img_generic_frame", "new_img_generic_frame_strengths"),
        )
        for row in rows
    ]
    oracle_strong_pairs = [
        (
            strong_gold_frames(row, "img"),
            strong_pred_frames(oracle_by_uuid.get(row["uuid"], {}), "new_img_generic_frame", "new_img_generic_frame_strengths"),
        )
        for row in rows
    ]
    strong_settings = [
        (f"Text — {TEXT_MODEL_ID}", compare_pairs(text_strong_pairs)),
        (f"Image, no oracle — {VLM_MODEL_ID}", compare_pairs(no_oracle_strong_pairs)),
        (f"Image, with oracle — {VLM_MODEL_ID}", compare_pairs(oracle_strong_pairs)),
    ]
    lines.append("\n## Summary — strong framings only\n")
    lines.append(
        "Same comparison, restricted to high-confidence frames on both sides: a gold frame counts "
        "only if >=2 of the 3 ensemble models rated it \"strong\" (not \"moderate\"), and a baseline "
        "prediction counts only where the baseline itself rated that frame \"strong\".\n"
    )
    lines.append(summary_table(strong_settings, n))

    for name, s in settings:
        lines.append(f"\n## {name}\n")
        lines.append("**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. "
                      "**most over-predicted** (baseline predicted, not in ground truth — false positives):\n")
        lines.append("| Missed | Rows | | Over-predicted | Rows |")
        lines.append("|---|---:|---|---|---:|")
        missed_list = s["missed"].most_common(8)
        extra_list = s["extra"].most_common(8)
        for i in range(max(len(missed_list), len(extra_list))):
            m = missed_list[i] if i < len(missed_list) else ("", "")
            e = extra_list[i] if i < len(extra_list) else ("", "")
            m_name = label_name(m[0]) if m[0] else ""
            e_name = label_name(e[0]) if e[0] else ""
            lines.append(f"| {m_name} | {m[1]} | | {e_name} | {e[1]} |")

    # Does oracle help, or just get copied?
    no_oracle_f1s = no_oracle_stats["per_row_f1"]
    oracle_f1s = oracle_stats["per_row_f1"]
    improved = sum(1 for a, b in zip(no_oracle_f1s, oracle_f1s) if b > a)
    worsened = sum(1 for a, b in zip(no_oracle_f1s, oracle_f1s) if b < a)
    unchanged = n - improved - worsened

    copy_count, oracle_nonempty = 0, 0
    for row in rows:
        oracle_text = set(row.get("consolidated_text_generic_frame") or [])
        oracle_img_pred = set((oracle_by_uuid.get(row["uuid"], {}) or {}).get("new_img_generic_frame") or [])
        if oracle_text:
            oracle_nonempty += 1
            if oracle_img_pred == oracle_text:
                copy_count += 1

    lines.append("\n## Does the oracle text frame help image prediction, or just get copied?\n")
    lines.append("| | No oracle | With oracle |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Precision | {no_oracle_stats['precision']:.2f} | {oracle_stats['precision']:.2f} |")
    lines.append(f"| Recall | {no_oracle_stats['recall']:.2f} | {oracle_stats['recall']:.2f} |")
    lines.append(f"| F1 | {no_oracle_stats['f1']:.2f} | {oracle_stats['f1']:.2f} |")
    lines.append("")
    lines.append("| Outcome of adding the oracle (per-row F1 change) | Rows |")
    lines.append("|---|---:|")
    lines.append(f"| Improved agreement | {improved}/{n} ({improved/n:.0%}) |")
    lines.append(f"| Worsened agreement | {worsened}/{n} ({worsened/n:.0%}) |")
    lines.append(f"| Unchanged | {unchanged}/{n} ({unchanged/n:.0%}) |")
    lines.append("")
    lines.append(
        f"Of {oracle_nonempty} rows with a non-empty oracle text frame, the oracle-setting image prediction "
        f"was an **exact copy** of the text frame in **{copy_count} ({copy_count/oracle_nonempty:.0%})** — "
        f"a high rate here suggests the model leans on the given text label rather than looking at the image."
    )

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
