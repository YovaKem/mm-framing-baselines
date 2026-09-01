"""
Baseline: Qwen3-VL-4B-Instruct zero-shot image-frame classification, run
locally on GPU — a small open-weight VLM's take on the image-framing task,
compared against the consolidated ground truth (scripts/report_baselines.py).

Both settings give the model the image PLUS the full article title+text as
context (the model reads the article to understand what's going on, but
still judges frames specifically as conveyed by the image — see
IMAGE_SYSTEM_PROMPT). Controlled by --oracle:
  --no-oracle (default): image + article title + text — the model must infer
    for itself whether the image's framing lines up with the text's.
  --oracle: image + article title + text + the oracle text-frame labels for
    that document, given explicitly as known context — isolates whether
    handing over the text-frame label (beyond what the model could infer
    itself from the text) changes/improves its image framing, versus just
    seeing the image + article text.

Reads directly from the participant-facing package built by
scripts/package_subtask2.py (data/subtask2_<split>/), NOT from
sample_consolidated.jsonl — i.e. exactly the article `.txt` + image `.jpg`
pairs (and, for --oracle, oracle_text_frames.tsv) a real participant would
receive, rather than this project's own internal data file. This is
deliberate: it keeps the baseline honest about what a participant's system
can actually see.

Reuses IMAGE_SYSTEM_PROMPT from scripts/relabel_frames.py for a fair
comparison — the paper's own image-framing prompt, no strength grading, just
frame presence + a reason.

Usage:
    python scripts/baseline_qwen_vlm.py --oracle
    python scripts/baseline_qwen_vlm.py --no-oracle
    python scripts/baseline_qwen_vlm.py --oracle --limit 3
    python scripts/baseline_qwen_vlm.py --split train_lora --no-oracle
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from common import (  # noqa: E402
    CANONICAL_LABEL_TO_KEY,
    DATA_DIR,
    extract_json_object,
    strip_none_key,
    write_jsonl,
)
from package_subtask2 import SPLITS, doc_id_to_uuid, parse_article_txt  # noqa: E402
from relabel_frames import IMAGE_SYSTEM_PROMPT, MAX_ARTICLE_CHARS  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def baseline_vlm_path(oracle):
    suffix = "with_oracle" if oracle else "no_oracle"
    return DATA_DIR / f"baseline_qwen3-vl-4b-instruct_{suffix}.jsonl"


def package_dir(split):
    return DATA_DIR / f"subtask2_{split.replace('_lora', '')}"


def load_oracle_labels(split):
    """doc-id (with .txt extension) -> list of frame label strings, merging any
    rows a document's labels were split across (see package_subtask2.py's
    oracle_rows). Columns are <file-name> <start> <end> <framing-dimension>*."""
    path = package_dir(split) / "oracle_text_frames.tsv"
    labels_by_doc = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        cols = line.split("\t")
        doc_txt_name, labels = cols[0], cols[3:]
        labels_by_doc.setdefault(doc_txt_name, []).extend(labels)
    return labels_by_doc


def parse_response(text):
    try:
        parsed = extract_json_object(text)
        names = parsed.get("frames-list", [])
        keys, unrecognized = [], []
        for name in names:
            key = CANONICAL_LABEL_TO_KEY.get(str(name).strip().lower())
            if key is None:
                unrecognized.append(name)
            else:
                keys.append(key)
        return {
            "new_img_generic_frame": strip_none_key(keys),
            "new_img_generic_frame_exp": parsed.get("reason", ""),
            "new_img_generic_frame_error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "new_img_generic_frame": [],
            "new_img_generic_frame_exp": None,
            "new_img_generic_frame_error": f"parse_failed: {e} | raw={text[:200]!r}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", dest="oracle", action="store_true")
    parser.add_argument("--no-oracle", dest="oracle", action="store_false")
    parser.set_defaults(oracle=False)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--max-new-tokens", type=int, default=450)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (for smoke testing)")
    args = parser.parse_args()

    data_dir = package_dir(args.split)
    if not data_dir.exists():
        raise SystemExit(f"{data_dir} not found — run scripts/package_subtask2.py --split {args.split} first")
    doc_ids = sorted(p.stem for p in data_dir.glob("*.txt"))
    if args.limit:
        doc_ids = doc_ids[: args.limit]

    oracle_labels_by_doc = load_oracle_labels(args.split) if args.oracle else {}

    print(f"Loading {MODEL_ID} (oracle={args.oracle}, split={args.split}) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    for doc_id in tqdm(doc_ids, desc=f"Qwen3-VL-4B-Instruct image baseline (oracle={args.oracle})"):
        title, article_text = parse_article_txt(data_dir / f"{doc_id}.txt")
        image = Image.open(data_dir / f"{doc_id}.jpg").convert("RGB")

        text_parts = [f"TITLE: {title}\n\nTEXT:\n{article_text[:MAX_ARTICLE_CHARS]}"]
        if args.oracle:
            oracle_labels = oracle_labels_by_doc.get(f"{doc_id}.txt") or ["(none — no text frame applies)"]
            text_parts.append(f"KNOWN TEXT FRAME(S) FOR THIS ARTICLE (ground truth): {', '.join(oracle_labels)}")
        user_text = "\n".join(text_parts)

        messages = [
            {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        response_text = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        result = {"uuid": doc_id_to_uuid(doc_id)}
        result.update(parse_response(response_text))
        results.append(result)

    errors = sum(1 for r in results if r["new_img_generic_frame_error"])
    if args.limit:
        print(f"\n[--limit smoke test, nothing written] {len(results)} rows, {errors} parse failures.")
        for r in results:
            print(r)
    else:
        out_path = baseline_vlm_path(args.oracle)
        write_jsonl(out_path, results)
        print(f"\nDone. {len(results)} rows, {errors} parse failures.")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
