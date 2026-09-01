"""
Baseline: Qwen3-4B-Instruct-2507 zero-shot text-frame classification, run
locally on GPU (not via OpenRouter) — a small open-weight model's take on the
same task the 3-model API ensemble did, for comparison against the
consolidated ground truth (scripts/report_baselines.py).

Reuses the exact same taxonomy/prompt (TEXT_SYSTEM_PROMPT) as
scripts/relabel_frames.py for a fair comparison — the paper's own
text-framing prompt, no strength grading, just frame presence + a reason.

Reads directly from the participant-facing package built by
scripts/package_subtask1.py (data/subtask1_<split>/), NOT from
sample_news.jsonl/sample_consolidated.jsonl — i.e. exactly the article `.txt`
files a real participant would receive, rather than this project's own
internal data file (same rationale as baseline_qwen_vlm.py's Subtask 2 read).

Usage:
    python scripts/baseline_qwen_text.py
    python scripts/baseline_qwen_text.py --max-new-tokens 400
    python scripts/baseline_qwen_text.py --split train_lora
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import (  # noqa: E402
    CANONICAL_LABEL_TO_KEY,
    extract_json_object,
    relabel_model_path,
    strip_none_key,
    write_jsonl,
)
from package_subtask1 import package_dir  # noqa: E402
from package_subtask2 import SPLITS, doc_id_to_uuid, parse_article_txt  # noqa: E402
from relabel_frames import MAX_ARTICLE_CHARS, TEXT_SYSTEM_PROMPT  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


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
            "new_text_generic_frame": strip_none_key(keys),
            "new_text_generic_frame_exp": parsed.get("reason", ""),
            "new_text_generic_frame_error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "new_text_generic_frame": [],
            "new_text_generic_frame_exp": None,
            "new_text_generic_frame_error": f"parse_failed: {e} | raw={text[:200]!r}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (for smoke testing)")
    args = parser.parse_args()

    data_dir = package_dir(args.split)
    if not data_dir.exists():
        raise SystemExit(f"{data_dir} not found — run scripts/package_subtask1.py --split {args.split} first")
    doc_ids = sorted(p.stem for p in data_dir.glob("*.txt"))
    if args.limit:
        doc_ids = doc_ids[: args.limit]

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    for doc_id in tqdm(doc_ids, desc=f"Qwen3-4B-Instruct text baseline"):
        title, article_text = parse_article_txt(data_dir / f"{doc_id}.txt")
        user_content = f"TITLE: {title}\n\nTEXT:\n{article_text[:MAX_ARTICLE_CHARS]}"
        messages = [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        result = {"uuid": doc_id_to_uuid(doc_id)}
        result.update(parse_response(response_text))
        results.append(result)

    errors = sum(1 for r in results if r["new_text_generic_frame_error"])
    if args.limit:
        print(f"\n[--limit smoke test, nothing written] {len(results)} rows, {errors} parse failures.")
        for r in results:
            print(r)
    else:
        out_path = relabel_model_path(MODEL_ID)
        write_jsonl(out_path, results)
        print(f"\nDone. {len(results)} rows, {errors} parse failures.")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
