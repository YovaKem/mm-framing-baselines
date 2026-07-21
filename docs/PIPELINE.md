# Pipeline reference

How to reproduce or rerun any step of the process described in the main
[README](../README.md). Run from the project root, in order.

## Setup

```bash
pip install -r requirements.txt
```

Labeling and filtering call an LLM via [OpenRouter](https://openrouter.ai). Put
your key in a local `.env` (git-ignored, never commit it):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Steps

```bash
# Sample 300 articles, scraping each one's real article text and lead image
# from its source URL (the dataset only stores derived labels, not content).
python scripts/build_sample.py

# Column-by-column stats + observed frame-tag frequency for the sample.
python scripts/inspect_columns.py

# Drop non-news rows (game announcements, listicles, lifestyle content).
python scripts/filter_news.py

# Label every article's text frame(s) and image frame(s), zero-shot, once per
# ensemble model.
python scripts/relabel_frames.py --model anthropic/claude-haiku-4.5
python scripts/relabel_frames.py --model openai/gpt-5.4-mini
python scripts/relabel_frames.py --model google/gemini-3.5-flash

# Take the 2-of-3 consensus across the three models as the target label, for
# text and image separately. Each model's individual output is kept alongside
# for transparency.
python scripts/consolidate_annotations.py

# Flag rows worth a closer manual look (low model agreement, a label call
# failing, etc.).
python scripts/flag_issues.py

# Local baselines (see "Local model setup" below): Qwen3-4B-Instruct
# zero-shot text framing, and Qwen3-VL-4B-Instruct zero-shot image framing in
# two settings, both compared against the consensus target.
python scripts/baseline_qwen_text.py
python scripts/baseline_qwen_vlm.py --no-oracle
python scripts/baseline_qwen_vlm.py --oracle
python scripts/report_baselines.py
```

To only redo the image-frame labeling for an ensemble model (reusing its
existing text results), use `relabel_frames.py --model <name> --image-only`.

## Local model setup

The two Qwen baselines run locally on GPU rather than through OpenRouter, and
need a newer `transformers`/`torch` than `requirements.txt` pins — they live in
their own venv:

```bash
python3 -m venv .venv
# If your root filesystem is small, redirect pip's cache/temp elsewhere first:
export PIP_CACHE_DIR=/workspace/pip_cache TMPDIR=/workspace/tmp

.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install numpy "transformers>=4.57" accelerate qwen-vl-utils pillow tqdm openai python-dotenv

HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_text.py
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_vlm.py --no-oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_vlm.py --oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/report_baselines.py
```

Both baseline scripts accept `--limit N` for a quick smoke test. `HF_HOME`
controls where model weights are cached — point it somewhere with real disk
space.

## Script reference

| Script | Role |
|---|---|
| `scripts/common.py` | Shared paths, the 15-category frame taxonomy, parsing helpers |
| `scripts/build_sample.py` | Sample + scrape article text/images |
| `scripts/inspect_columns.py` | Column stats and frame-tag frequency |
| `scripts/filter_news.py` | Drop non-news rows |
| `scripts/relabel_frames.py` | Zero-shot text/image frame labeling (one ensemble model per run) |
| `scripts/consolidate_annotations.py` | 2-of-3 consensus across the ensemble |
| `scripts/flag_issues.py` | Flag rows worth manual review |
| `scripts/baseline_qwen_text.py` | Local Qwen3-4B-Instruct text baseline |
| `scripts/baseline_qwen_vlm.py` | Local Qwen3-VL-4B-Instruct image baseline (`--oracle` / `--no-oracle`) |
| `scripts/report_baselines.py` | Baselines vs. consensus target, with paper-matching metrics |

## Sampling caveat

Because sampling keeps drawing rows until 300 are fully scrapeable, the sample
is conditioned on being scrapeable — biased toward outlets/links still live and
unpaywalled two years on — not a strict random sample of the full dataset. Every
attempt (kept or rejected, and why) is logged to
`data/scrape_attempts_log.jsonl`. Filtering then further reduces the count by
dropping non-news rows, with no replenishment.
