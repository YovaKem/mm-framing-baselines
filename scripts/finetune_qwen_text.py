"""
Baseline: LoRA-finetune Qwen3-4B-Instruct on the split=train_lora rows (target:
consolidated_text_generic_frame, the 2-of-3 ensemble consensus — no human
text-frame ground truth exists), then run it on the same split=test rows as
the zero-shot baseline (baseline_qwen_text.py) for a direct, apples-to-apples
comparison in scripts/report_baselines.py.

Same TEXT_SYSTEM_PROMPT/input format as the zero-shot text baseline. Trains a
small LoRA adapter (not a full finetune) — reasonable given the training set
is only ~100 rows. Simple per-example SFT loop (not transformers.Trainer):
loss is masked to the assistant's JSON completion only (prompt tokens
excluded), teacher-forced on the ground-truth frames-list.

Usage:
    python scripts/finetune_qwen_text.py
    python scripts/finetune_qwen_text.py --epochs 3 --lr 1e-4
"""
import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from baseline_qwen_text import MODEL_ID, parse_response  # noqa: E402
from common import (  # noqa: E402
    CANONICAL_FRAMES,
    CONSOLIDATED_PATH,
    DATA_DIR,
    RANDOM_SEED,
    read_jsonl,
    write_jsonl,
)
from relabel_frames import MAX_ARTICLE_CHARS, TEXT_SYSTEM_PROMPT  # noqa: E402

ADAPTER_DIR = DATA_DIR / "lora_adapters" / "qwen3-4b-instruct-text"
OUT_PATH = DATA_DIR / "baseline_qwen3-4b-instruct-lora_text.jsonl"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def target_json(frame_keys):
    names = [CANONICAL_FRAMES[k].split(" — ")[0] for k in frame_keys] or ["None"]
    return json.dumps({"frames-list": names, "reason": "Determined from the article's content and framing choices."})


def user_content(row):
    article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]
    return f"TITLE: {row.get('title', '')}\n\nTEXT:\n{article_text}"


def build_example(tokenizer, row, device):
    prompt_messages = [
        {"role": "system", "content": TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content(row)},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": target_json(row.get("consolidated_text_generic_frame") or [])}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        full_messages, add_generation_prompt=False, tokenize=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return full_ids.to(device), labels.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not CONSOLIDATED_PATH.exists():
        raise SystemExit(f"{CONSOLIDATED_PATH} not found — run scripts/consolidate_annotations.py first")
    rows = read_jsonl(CONSOLIDATED_PATH)
    train_rows = [r for r in rows if r.get("split") == "train_lora"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    print(f"Training on {len(train_rows)} rows, evaluating on {len(test_rows)} rows.")

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False  # required alongside gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # needed for grad checkpointing through a frozen base model
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    rng = random.Random(args.seed)
    model.train()
    for epoch in range(args.epochs):
        order = list(train_rows)
        rng.shuffle(order)
        total_loss = 0.0
        for row in tqdm(order, desc=f"LoRA finetune (text) epoch {epoch + 1}/{args.epochs}"):
            input_ids, labels = build_example(tokenizer, row, model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch + 1}: avg loss = {total_loss / len(order):.4f}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")

    model.eval()
    model.config.use_cache = True
    results = []
    for row in tqdm(test_rows, desc="Qwen3-4B-Instruct + LoRA text baseline (eval)"):
        messages = [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content(row)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = {"uuid": row["uuid"]}
        result.update(parse_response(response_text))
        results.append(result)

    errors = sum(1 for r in results if r["new_text_generic_frame_error"])
    write_jsonl(OUT_PATH, results)
    print(f"\nDone. {len(results)} rows, {errors} parse failures.")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
