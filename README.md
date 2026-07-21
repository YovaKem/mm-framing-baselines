# mm-framing-baselines

Baselines and manual-validation tooling for [`copenlu/mm-framing`](https://huggingface.co/datasets/copenlu/mm-framing),
the dataset behind [*Multi-Modal Framing Analysis of News*](https://arxiv.org/abs/2503.20960)
(Arora et al.). See [`docs/DATASET.md`](docs/DATASET.md) for a full column/taxonomy
reference and important caveats about what the dataset does and doesn't contain.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

Run in order from the project root:

```bash
# 1. Draw a reproducible random sample of 300 rows (seed 42) from the
#    paper's own filtered valid_framing_subset split.
python scripts/sample_dataset.py

# 2. Best-effort scrape of each row's original article text + lead image.
#    Live links from 2023-2024 will have real rot — failures are recorded,
#    not silently dropped.
python scripts/scrape_articles.py

# 3. Column-by-column stats + observed frame-tag frequency, written to
#    data/inspection_report.md and printed to stdout.
python scripts/inspect_columns.py

# 4. Structural sweep flagging likely data-quality issues per row
#    (missing/malformed fields, out-of-taxonomy tags, thin explanations,
#    scrape failures) — written to data/flags.json.
python scripts/flag_issues.py

# 5. Manual validation UI, defaults to showing only flagged rows first.
streamlit run app/validate_app.py
```

Your validation decisions are saved incrementally to `data/validation_results.csv`
(safe to stop and resume the Streamlit app at any time).

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
  sample_dataset.py   step 1
  scrape_articles.py  step 2
  inspect_columns.py  step 3
  flag_issues.py      step 4
app/
  validate_app.py     step 5 — Streamlit validation UI
docs/
  DATASET.md          dataset schema, taxonomy, and paper notes
data/                  generated artifacts (raw sample + images gitignored,
                       flags/report/validation results tracked)
```
