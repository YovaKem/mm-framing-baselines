# mm-framing-baselines

Baselines and manual-validation tooling for [`copenlu/mm-framing`](https://huggingface.co/datasets/copenlu/mm-framing),
the dataset behind [*Multi-Modal Framing Analysis of News*](https://arxiv.org/abs/2503.20960)
(Arora et al.). See [`docs/DATASET.md`](docs/DATASET.md) for a full column/taxonomy
reference and important caveats about what the dataset does and doesn't contain.

## Setup

```bash
pip install -r requirements.txt
```

The semantic frame-accuracy check (step 3 below) calls a vision LLM via
[OpenRouter](https://openrouter.ai). Put your key in a local `.env` (already
git-ignored, never commit it):

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

# 3. Semantic accuracy check: shows a vision LLM (anthropic/claude-haiku-4.5
#    via OpenRouter) the real article text + image for each row, alongside
#    the text-generic-frame / img-generic-frame labels and the original
#    model's justification, and asks it to independently judge whether each
#    label is actually well-supported. Written to data/frame_judgments.json.
python scripts/judge_frames.py

# 4. Merges the semantic judgments with structural checks (missing/malformed
#    fields, out-of-taxonomy tags, thin explanations) into data/flags.json.
python scripts/flag_issues.py

# 5. Manual validation UI — shows the article text + image + both frame
#    labels + the automated judge's verdict and reasoning per row, defaults
#    to showing only flagged rows first.
streamlit run app/validate_app.py
```

Your validation decisions are saved incrementally to `data/validation_results.csv`
(safe to stop and resume the Streamlit app at any time).

**Sampling caveat:** because step 1 keeps drawing rows until 300 are fully
scrapeable, the final sample is a random draw *conditioned on being
scrapeable* — biased toward outlets/links still live and unpaywalled two years
later — not a strict random sample of the full dataset. Every attempt (kept or
rejected, and why) is logged to `data/scrape_attempts_log.jsonl` for
transparency.

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
  common.py           shared paths, the 15-category frame taxonomy, parsing helpers
  build_sample.py     step 1 — oversample + scrape until 300 complete rows
  inspect_columns.py  step 2
  judge_frames.py     step 3 — LLM semantic frame-accuracy check
  flag_issues.py      step 4
app/
  validate_app.py     step 5 — Streamlit validation UI
docs/
  DATASET.md          dataset schema, taxonomy, and paper notes
data/                 generated artifacts (raw sample/images/attempt-log
                      gitignored, flags/judgments/report/validation results tracked)
```
