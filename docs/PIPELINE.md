# Pipeline reference

How to reproduce or rerun any step of the process described in the main
[README](../README.md). Run from the project root, in order.

## Setup

```bash
pip install -r requirements.txt
```

Labeling calls an LLM via [OpenRouter](https://openrouter.ai). Put your key
in a local `.env` (git-ignored, never commit it):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Steps

```bash
# Join data/human_image_frame_labels.csv (the paper's own human
# double-annotation pass over 600 images) against the full copenlu/mm-framing
# dataset to recover url/title/metadata, scrape article text (with a Wayback
# Machine fallback for dead/blocked links), and match each uuid to its image
# (extracted ahead of time from the provided images.zip into data/images/).
# Assigns a fixed split=test/train_lora once scraping is done. No
# news/genuine-journalism filtering is applied to this sample.
python scripts/build_human_sample.py

# Label every article's TEXT frame(s), zero-shot, once per ensemble model —
# the silver-standard text target (no human text-frame annotation exists).
python scripts/relabel_frames.py --model anthropic/claude-haiku-4.5
python scripts/relabel_frames.py --model openai/gpt-5.4-mini
python scripts/relabel_frames.py --model google/gemini-3.5-flash

# Consolidate: text frames get the 2-of-3 ensemble consensus; image frames
# get the human ground truth straight from the CSV (not a vote). Each
# ensemble model's individual text output is kept alongside for transparency.
python scripts/consolidate_annotations.py

# Flag rows worth a closer manual look (low text-ensemble agreement, a
# relabel call failing, image frame not a subset of text frame, etc.).
python scripts/flag_issues.py

# Local baselines (see "Local model setup" below), all scored on the same
# split=test rows: Qwen3-4B-Instruct zero-shot + LoRA-finetuned text framing,
# and Qwen3-VL-4B-Instruct zero-shot + LoRA-finetuned image framing in two
# settings (oracle / no-oracle). The two finetunes train on split=train_lora.
python scripts/baseline_qwen_text.py
python scripts/baseline_qwen_vlm.py --no-oracle
python scripts/baseline_qwen_vlm.py --oracle
python scripts/finetune_qwen_text.py
python scripts/finetune_qwen_vlm.py --no-oracle
python scripts/finetune_qwen_vlm.py --oracle
python scripts/report_baselines.py
```

## Local model setup

The baseline/finetune scripts run locally on GPU rather than through
OpenRouter, and need a newer `transformers`/`torch` than `requirements.txt`
pins — they live in their own venv:

```bash
python3 -m venv .venv
# If your root filesystem is small, redirect pip's cache/temp elsewhere first:
export PIP_CACHE_DIR=/workspace/pip_cache TMPDIR=/workspace/tmp

.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install numpy "transformers>=4.57" accelerate peft qwen-vl-utils pillow tqdm openai python-dotenv

HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_text.py
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_vlm.py --no-oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_vlm.py --oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/finetune_qwen_text.py
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/finetune_qwen_vlm.py --no-oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/finetune_qwen_vlm.py --oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/report_baselines.py
```

The zero-shot baseline scripts accept `--limit N` for a quick smoke test.
`HF_HOME` controls where model weights are cached — point it somewhere with
real disk space. LoRA adapters are saved under `data/lora_adapters/`
(git-ignored, regenerate by rerunning the finetune scripts).

## Script reference

| Script | Role |
|---|---|
| `scripts/common.py` | Shared paths, the 15-category frame taxonomy (paper-matching: 14 real frames + "None"), parsing helpers |
| `scripts/build_human_sample.py` | Join the human image-label CSV against the full HF dataset, scrape article text, match images, assign the test/train_lora split |
| `scripts/inspect_columns.py` | Column stats and frame-tag frequency (legacy, ran on the old 300-article sample) |
| `scripts/relabel_frames.py` | Zero-shot TEXT frame labeling (one ensemble model per run); also defines the shared TEXT/IMAGE system prompts every baseline/finetune script imports |
| `scripts/consolidate_annotations.py` | 2-of-3 consensus for text; carries the human label through as-is for image |
| `scripts/flag_issues.py` | Flag rows worth manual review |
| `scripts/baseline_qwen_text.py` | Local Qwen3-4B-Instruct zero-shot text baseline |
| `scripts/baseline_qwen_vlm.py` | Local Qwen3-VL-4B-Instruct zero-shot image baseline (`--oracle` / `--no-oracle`) |
| `scripts/finetune_qwen_text.py` | LoRA-finetune Qwen3-4B-Instruct on split=train_lora, eval on split=test |
| `scripts/finetune_qwen_vlm.py` | LoRA-finetune Qwen3-VL-4B-Instruct on split=train_lora (`--oracle` / `--no-oracle`), eval on split=test |
| `scripts/report_baselines.py` | All 6 baselines vs. ground truth, with paper-matching metrics |

## Sampling caveat

`data/human_image_frame_labels.csv` has 600 uuids from the paper's own human
double-annotation pass over images; 578 have a matching row (url, title,
metadata) in the full `copenlu/mm-framing` dataset (22 have no match
anywhere and are dropped). Of those 578, only ~400 are actually scrapeable —
**CBS News alone accounts for 264/578 (46%) of this sample**, and CBS's WAF
blocks both live scraping and, largely, the Wayback Machine's own crawler
too. `build_human_sample.py` tries a live fetch first, then falls back to
the closest Wayback Machine snapshot (for both outright failures and
thin/paywalled extracts) before giving up on a row — every attempt is logged
to `data/scrape_attempts_log.jsonl` for transparency. The resulting sample is
conditioned on being scrapeable — biased toward outlets/links still live (or
at least archived) two years on — not a strict random sample of the CSV's
578 uuids. The test/train_lora split (300/100, originally targeted at
350/228 before this shortfall was discovered) is assigned only after
scraping completes, from whatever actually succeeded.
