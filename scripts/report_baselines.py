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


def compare(rows, gold_field, pred_by_uuid, pred_field):
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
    return {
        "n": n,
        "avg_jaccard": sum(jaccards) / n,
        "identical": identical,
        "disjoint": disjoint,
        "avg_gold_size": sum(gold_sizes) / n,
        "avg_pred_size": sum(pred_sizes) / n,
        "missed": missed,
        "extra": extra,
        "jaccards": jaccards,
    }


def top_frames_table(counter, n_rows):
    lines = ["| Frame | Rows |", "|---|---:|"]
    for key, count in counter.most_common(8):
        lines.append(f"| {label_name(key)} | {count} |")
    if not counter:
        lines.append("| _(none)_ | |")
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

    lines = [f"# Baseline report\n", f"Rows: {n}\n"]

    lines.append("## Summary\n")
    lines.append("| Baseline | Avg Jaccard | Identical | Disjoint | Avg labels (gold) | Avg labels (pred) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, s in settings:
        lines.append(
            f"| {name} | {s['avg_jaccard']:.2f} | {s['identical']}/{n} ({s['identical']/n:.0%}) | "
            f"{s['disjoint']}/{n} ({s['disjoint']/n:.0%}) | {s['avg_gold_size']:.2f} | {s['avg_pred_size']:.2f} |"
        )

    for name, s in settings:
        lines.append(f"\n## {name}\n")
        lines.append("**Most-missed** (in ground truth, baseline didn't predict) vs. **most over-predicted** (baseline predicted, not in ground truth):\n")
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
    no_oracle_jaccards = no_oracle_stats["jaccards"]
    oracle_jaccards = oracle_stats["jaccards"]
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

    lines.append("\n## Does the oracle text frame help image prediction, or just get copied?\n")
    lines.append("| | No oracle | With oracle |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Avg Jaccard vs. ground truth | {sum(no_oracle_jaccards)/n:.2f} | {sum(oracle_jaccards)/n:.2f} |")
    lines.append("")
    lines.append("| Outcome of adding the oracle | Rows |")
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
