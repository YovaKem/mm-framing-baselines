"""
Baseline: LoRA-finetune Qwen3-VL-4B-Instruct on the split=train_lora rows
(target: consolidated_img_generic_frame — the paper's own human
double-annotation, real ground truth), then run it on the same split=test
rows as the zero-shot baseline (baseline_qwen_vlm.py) for a direct,
apples-to-apples comparison in scripts/report_baselines.py.

Same two settings as the zero-shot image baseline, --oracle / --no-oracle,
mirrored exactly for training too: the oracle setting's input includes an
extra "KNOWN TEXT FRAME(S)" context line the no-oracle one doesn't, so this
trains and evaluates a SEPARATE adapter per setting (not one adapter run
twice) — matches how the zero-shot baseline treats these as genuinely
different inputs, not just different eval-time prompts.

Simple per-example SFT loop (not transformers.Trainer): loss is masked to the
assistant's JSON completion only, teacher-forced on the ground-truth
frames-list.

Usage:
    python scripts/finetune_qwen_vlm.py --no-oracle
    python scripts/finetune_qwen_vlm.py --oracle
"""
import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
# Images vary in resolution -> vision-tower activation tensors vary in size step to
# step, which fragments the CUDA allocator over a training run (see PyTorch's own
# OOM message suggesting exactly this). Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from baseline_qwen_vlm import MODEL_ID, parse_response  # noqa: E402
from common import (  # noqa: E402
    CANONICAL_FRAMES,
    CONSOLIDATED_PATH,
    DATA_DIR,
    RANDOM_SEED,
    read_jsonl,
    write_jsonl,
)
from relabel_frames import IMAGE_SYSTEM_PROMPT, MAX_ARTICLE_CHARS  # noqa: E402

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def adapter_dir(oracle):
    return DATA_DIR / "lora_adapters" / f"qwen3-vl-4b-instruct-image-{'oracle' if oracle else 'no_oracle'}"


def out_path(oracle):
    suffix = "with_oracle" if oracle else "no_oracle"
    return DATA_DIR / f"baseline_qwen3-vl-4b-instruct-lora_{suffix}.jsonl"


def target_json(frame_keys):
    names = [CANONICAL_FRAMES[k].split(" — ")[0] for k in frame_keys] or ["None"]
    return json.dumps({"frames-list": names, "reason": "Determined from the image's visual content."})


def user_text(row, oracle, canonical_labels):
    article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]
    parts = [f"TITLE: {row.get('title', '')}\n\nTEXT:\n{article_text}"]
    if oracle:
        oracle_keys = row.get("consolidated_text_generic_frame") or []
        oracle_labels = [canonical_labels.get(k, k) for k in oracle_keys] or ["(none — no text frame applies)"]
        parts.append(f"KNOWN TEXT FRAME(S) FOR THIS ARTICLE (ground truth): {', '.join(oracle_labels)}")
    return "\n".join(parts)


def build_example(processor, row, oracle, canonical_labels, device):
    image = Image.open(DATA_DIR / row["image_local_path"]).convert("RGB")
    text = user_text(row, oracle, canonical_labels)
    prompt_messages = [
        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]
    target = target_json(row.get("consolidated_img_generic_frame") or [])
    full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]

    prompt_inputs = processor.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    full_inputs = processor.apply_chat_template(
        full_messages, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt"
    )
    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return {k: v.to(device) for k, v in full_inputs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", dest="oracle", action="store_true")
    parser.add_argument("--no-oracle", dest="oracle", action="store_false")
    parser.set_defaults(oracle=False)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=450)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not CONSOLIDATED_PATH.exists():
        raise SystemExit(f"{CONSOLIDATED_PATH} not found — run scripts/consolidate_annotations.py first")
    rows = read_jsonl(CONSOLIDATED_PATH)
    train_rows = [r for r in rows if r.get("split") == "train_lora"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    print(f"Training on {len(train_rows)} rows, evaluating on {len(test_rows)} rows (oracle={args.oracle}).")

    canonical_labels = {k: v.split(" — ")[0] for k, v in CANONICAL_FRAMES.items()}

    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False  # required alongside gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # needed for grad checkpointing through a frozen base model
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    rng = random.Random(args.seed)
    model.train()
    for epoch in range(args.epochs):
        order = list(train_rows)
        rng.shuffle(order)
        total_loss = 0.0
        desc = f"LoRA finetune (image, oracle={args.oracle}) epoch {epoch + 1}/{args.epochs}"
        for row in tqdm(order, desc=desc):
            inputs = build_example(processor, row, args.oracle, canonical_labels, model.device)
            outputs = model(**inputs)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
            del inputs, outputs
            torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1}: avg loss = {total_loss / len(order):.4f}")

    out_dir = adapter_dir(args.oracle)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"Saved LoRA adapter to {out_dir}")

    model.eval()
    model.config.use_cache = True
    results = []
    for row in tqdm(test_rows, desc=f"Qwen3-VL-4B-Instruct + LoRA image baseline (eval, oracle={args.oracle})"):
        image = Image.open(DATA_DIR / row["image_local_path"]).convert("RGB")
        text = user_text(row, args.oracle, canonical_labels)
        messages = [
            {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        response_text = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = {"uuid": row["uuid"]}
        result.update(parse_response(response_text))
        results.append(result)

    errors = sum(1 for r in results if r["new_img_generic_frame_error"])
    write_jsonl(out_path(args.oracle), results)
    print(f"\nDone. {len(results)} rows, {errors} parse failures.")
    print(f"Wrote {out_path(args.oracle)}")


if __name__ == "__main__":
    main()
