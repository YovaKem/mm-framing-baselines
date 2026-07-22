"""
Baseline: Qwen3-4B-Instruct-2507 zero-shot text-frame classification, run
locally on GPU (not via OpenRouter) — a small open-weight model's take on the
same task the 3-model API ensemble did, for comparison against the
consolidated ground truth (scripts/report_baselines.py).

Reuses the exact same taxonomy/prompt (TEXT_SYSTEM_PROMPT) as
scripts/relabel_frames.py for a fair comparison — the paper's own
text-framing prompt, no strength grading, just frame presence + a reason.

Usage:
    python scripts/baseline_qwen_text.py
    python scripts/baseline_qwen_text.py --max-new-tokens 400
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import (  # noqa: E402
    CANONICAL_LABEL_TO_KEY,
    NEWS_SAMPLE_PATH,
    extract_json_object,
    read_jsonl,
    relabel_model_path,
    strip_none_key,
    write_jsonl,
)
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
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (for smoke testing)")
    args = parser.parse_args()

    if not NEWS_SAMPLE_PATH.exists():
        raise SystemExit(f"{NEWS_SAMPLE_PATH} not found — run scripts/build_human_sample.py first")
    rows = [r for r in read_jsonl(NEWS_SAMPLE_PATH) if r.get("split") == "test"]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    for row in tqdm(rows, desc=f"Qwen3-4B-Instruct text baseline"):
        article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]
        user_content = f"TITLE: {row.get('title', '')}\n\nTEXT:\n{article_text}"
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

        result = {"uuid": row["uuid"]}
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
