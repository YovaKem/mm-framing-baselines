# mm-framing-baselines

Baselines and manual-validation tooling for [`copenlu/mm-framing`](https://huggingface.co/datasets/copenlu/mm-framing),
the dataset behind [*Multi-Modal Framing Analysis of News*](https://arxiv.org/abs/2503.20960)
(Arora et al.). See [`docs/DATASET.md`](docs/DATASET.md) for a full column/taxonomy
reference and important caveats about what the dataset does and doesn't contain.

## Setup

```bash
pip install -r requirements.txt
```

Steps 3-5 below call an LLM via [OpenRouter](https://openrouter.ai). Put your key
in a local `.env` (already git-ignored, never commit it):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Pipeline

Run in order from the project root:

```bash
# 1. Build a 300-row sample where EVERY row has both article text and a lead
#    image successfully scraped. Draws from a seeded shuffle of the paper's
#    valid_framing_subset split and keeps pulling further into that order
#    until 300 rows clear both bars (roughly half of attempts fail due to
#    link rot / paywalls on these 2023-2024 articles — see caveat below).
python scripts/build_sample.py

# 2. Column-by-column stats + observed frame-tag frequency, written to
#    data/inspection_report.md and printed to stdout.
python scripts/inspect_columns.py

# 3. Drop rows that aren't genuine news/journalism (game announcements,
#    "best of" listicles, lifestyle content) — framing analysis assumes
#    editorial choices about a real-world issue, which doesn't apply to that
#    kind of content. Written to data/sample_news.jsonl.
python scripts/filter_news.py

# 4. Ignore the dataset's original labels and generate fresh text/image frame
#    labels from scratch with EACH of a 3-model ensemble (one call per model
#    per row, run separately): two independent LLM calls per row (text first,
#    then image), strong/moderate-only, with per-frame strength + explanation.
#    Written to data/relabel_<model-slug>.jsonl, one file per model.
python scripts/relabel_frames.py --model anthropic/claude-haiku-4.5
python scripts/relabel_frames.py --model openai/gpt-5.4-mini
python scripts/relabel_frames.py --model google/gemini-3.5-flash

# 5. Compare the single-model (claude-haiku-4.5) relabeling against the
#    original dataset labels (Jaccard overlap, most-added/dropped frames) and
#    report how often the new image frame set is a subset of the new text
#    frame set. Written to data/overlap_report.md.
python scripts/report_overlap.py

# 6. Consolidate the 3 models' independent label sets into one: a frame is
#    kept only if at least 2 of 3 models independently assigned it, for text
#    and image separately. Each model's raw output is kept alongside for
#    transparency. Written to data/sample_consolidated.jsonl +
#    data/consolidation_report.md.
python scripts/consolidate_annotations.py

# 7. Flag rows worth a closer look (a model's relabel call failing, low
#    ensemble agreement, consolidated image frames not a subset of text
#    frames, large disagreement vs. the original label, plus structural
#    checks on the original columns) into data/flags.json.
python scripts/flag_issues.py

# 8. Manual validation UI — shows the article text + image, the original
#    labels, the 2-of-3 consolidated labels, and each model's individual
#    labels (expandable), defaults to showing all rows.
streamlit run app/validate_app.py

# 9. Local small-model baselines (see "Local model baselines" below for the
#    one-time GPU venv setup): Qwen3-4B-Instruct zero-shot text framing, and
#    Qwen3-VL-4B-Instruct zero-shot image framing in two settings (blind vs.
#    given the ground-truth text frame as context), all compared against the
#    consolidated ensemble labels as ground truth.
python scripts/baseline_qwen_text.py
python scripts/baseline_qwen_vlm.py --no-oracle
python scripts/baseline_qwen_vlm.py --oracle
python scripts/report_baselines.py
```

Your validation decisions are saved incrementally to `data/validation_results.csv`
(safe to stop and resume the Streamlit app at any time).

**Sampling caveat:** because step 1 keeps drawing rows until 300 are fully
scrapeable, the final sample is a random draw *conditioned on being
scrapeable* — biased toward outlets/links still live and unpaywalled two years
later — not a strict random sample of the full dataset. Every attempt (kept or
rejected, and why) is logged to `data/scrape_attempts_log.jsonl` for
transparency. Step 3 then further reduces the count by dropping non-news rows
(no replenishment — the working set shrinks below 300).

## Local model baselines

Steps 9 (`baseline_qwen_text.py` / `baseline_qwen_vlm.py`) run two small
open-weight models locally on GPU — `Qwen/Qwen3-4B-Instruct-2507` (text) and
`Qwen/Qwen3-VL-4B-Instruct` (vision) — rather than through OpenRouter. These
need a newer `transformers`/`torch` than `requirements.txt` pins for the rest
of the project, so they live in their own venv to avoid disturbing anything
else in this environment:

```bash
python3 -m venv .venv
# Redirect pip's cache/temp off a small root disk if you have one (check `df -h`
# first — skip this export if your root filesystem has plenty of space):
export PIP_CACHE_DIR=/workspace/pip_cache TMPDIR=/workspace/tmp

.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install numpy "transformers>=4.57" accelerate qwen-vl-utils pillow tqdm openai python-dotenv

HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_text.py
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_vlm.py --no-oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/baseline_qwen_vlm.py --oracle
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/report_baselines.py
```

Both scripts accept `--limit N` for a quick smoke test before committing to a
full 238-row run, and `HF_HOME` just controls where model weights get cached
(point it somewhere with real disk space).

**Headline finding** (`data/baselines_report.md`): the text baseline agrees
with the consolidated ground truth about as well as any single ensemble
member (Jaccard 0.47). For images, giving the VLM the ground-truth text frame
as context ("oracle") made agreement *worse* (0.43) than showing it the image
alone (0.52) — and 54% of the time the oracle-setting prediction was an exact
copy of the given text label, suggesting the model leans on the handed-to-it
text frame rather than actually looking at the image.

## Viewing the Streamlit app over SSH

Streamlit runs an ordinary local web server on port 8501. To view it from your
own machine:

- **VS Code Remote-SSH**: the port is auto-forwarded — a popup / the "Ports" tab
  will offer a `localhost:8501` link.
- **Plain SSH**: add a tunnel, e.g. `ssh -L 8501:localhost:8501 <host>`, then open
  `http://localhost:8501` locally.

## Project layout

```
scripts/
  common.py               shared paths, the 15-category frame taxonomy, parsing helpers
  build_sample.py         step 1 — oversample + scrape until 300 complete rows
  inspect_columns.py      step 2
  filter_news.py          step 3 — drop non-news rows
  relabel_frames.py       step 4 — fresh LLM text/image frame labels (run once per ensemble model)
  report_overlap.py       step 5 — single-model-vs-original label comparison
  consolidate_annotations.py  step 6 — 2-of-3 majority-vote consolidation across the 3 models
  flag_issues.py          step 7
  baseline_qwen_text.py   step 9a — local Qwen3-4B-Instruct zero-shot text baseline
  baseline_qwen_vlm.py    step 9b — local Qwen3-VL-4B-Instruct zero-shot image baseline
  report_baselines.py     step 9c — baselines vs. consolidated ground truth
app/
  validate_app.py         step 8 — Streamlit validation UI
docs/
  DATASET.md              dataset schema, taxonomy, and paper notes
data/                     generated artifacts (raw sample/images/attempt-log
                          gitignored, everything else tracked)
```
