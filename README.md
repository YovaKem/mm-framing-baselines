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
#    labels from scratch: two independent LLM calls per row (text first, then
#    image), strong/moderate-only, with per-frame strength + explanation.
#    Stored side by side with the originals in data/sample_relabeled.jsonl.
python scripts/relabel_frames.py

# 5. Compare the fresh relabeling against the original dataset labels (Jaccard
#    overlap, most-added/dropped frames) and report how often the new image
#    frame set is a subset of the new text frame set. Written to
#    data/overlap_report.md.
python scripts/report_overlap.py

# 6. Flag rows worth a closer look (relabel-call failures, image frames not a
#    subset of text frames, large old-vs-new disagreement, plus structural
#    checks on the original columns) into data/flags.json.
python scripts/flag_issues.py

# 7. Manual validation UI — shows the article text + image, the original
#    labels, and the new labels side by side, defaults to flagged rows first.
streamlit run app/validate_app.py
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
  common.py             shared paths, the 15-category frame taxonomy, parsing helpers
  build_sample.py       step 1 — oversample + scrape until 300 complete rows
  inspect_columns.py    step 2
  filter_news.py        step 3 — drop non-news rows
  relabel_frames.py     step 4 — fresh LLM text/image frame labels, ignoring the originals
  report_overlap.py     step 5 — old-vs-new label comparison
  flag_issues.py        step 6
app/
  validate_app.py       step 7 — Streamlit validation UI
docs/
  DATASET.md            dataset schema, taxonomy, and paper notes
data/                   generated artifacts (raw sample/images/attempt-log
                        gitignored, everything else tracked)
```
