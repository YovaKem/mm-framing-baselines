# copenlu/mm-framing — dataset notes

Source: https://huggingface.co/datasets/copenlu/mm-framing
Paper: *Multi-Modal Framing Analysis of News*, Arora, Yadav, Antoniak, Belongie & Augenstein (arXiv:2503.20960)

## What "framing" means here

Framing theory studies how authors select which aspects of a topic to emphasize,
shaping how an audience interprets it. This paper's contribution is doing that
analysis at scale, and doing it for **both** the article text and its accompanying
photograph, using vision/language LLMs rather than manual qualitative coding.

## How the data was built

- ~500K US news articles, May 2023–Apr 2024, from 28 outlets spanning the political
  spectrum (left / left-lean / center / right-lean / right).
- Text frames labeled by Mistral-7B-Instruct-v0.3; image frames labeled by
  Pixtral-12B-2409 (both via vLLM).
- Validated against existing human-annotated data (Media Frames Corpus for text) and
  a small human double-annotation pass for images (600 images) and topics (190
  articles) done by the paper's authors — not full independent human annotation of
  the whole set. Treat every row as an **LLM prediction**, not ground truth (this
  project's own relabeling pass, below, treats it as exactly that).

## Splits

- `full` (`annotated_data.csv`, ~479K rows) — everything.
- `valid_framing_subset` (`framing_subset.csv`, ~154K rows) — the paper's own
  filtered, framing-analysis-ready subset: drops "None"/invalid frame predictions,
  articles under 100 words, and Sports/Media topics. **This is what
  `scripts/build_sample.py` samples from by default.**

## Important: the dataset stores labels, not source content

Neither the raw article text nor the image (or even an image URL) is stored as a
column. The only columns are metadata (`title`, `date_publish`, `source_domain`,
`political_leaning`, `url`) and the **LLM's output about** the text/image (labels +
free-text justifications). The only path back to the actual source content is
`url` — hence `scripts/build_sample.py`'s scraping step, which is a best-effort
live fetch and will have real link-rot for two-year-old articles.

## The 15-category generic frame taxonomy

Both `text-generic-frame` and `img-generic-frame` — **the two main labels this
project cares about** — draw from the same fixed, multi-label taxonomy, adapted
from Boydstun et al. (2014) / the Media Frames Corpus. Definitions below are the
codebook's own wording, verbatim — `scripts/relabel_frames.py` passes these full
definitions to the LLM annotator rather than relying on the surface label alone:

| Frame | Definition |
|---|---|
| Economic | Costs, benefits, or other financial implications |
| Capacity & Resources | Availability of physical, human or financial resources, and capacity of current systems |
| Morality | Religious or ethical implications, considerations, issues, etc. |
| Fairness & Equality | Balance or distribution of rights, responsibilities, and resources |
| Legality, Constitutionality & Jurisprudence | Rights, freedoms, and authority of individuals, corporations, and government |
| Policy Prescription & Evaluation | Discussion of specific policies aimed at addressing problems, needs, issues, etc. |
| Crime & Punishment | Effectiveness and implications of laws and their enforcement |
| Security & Defense | Threats to welfare of the individual, community, or nation |
| Health & Safety | Health care, sanitation, public safety |
| Quality of Life | Threats and opportunities for the individual's wealth, happiness, and well-being |
| Cultural Identity | Traditions, customs, or values of a social group in relation to a policy issue |
| Public Opinion | Attitudes and opinions of the general public, including polling and demographics |
| Political | Considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters |
| External Regulation & Reputation | International reputation or foreign policy of the U.S. |
| Other | Frames that do not fit into the above categories |

**Caveat about the raw CSV:** it stores short-form tags (e.g. `security`,
`legality`, `regulation`, `policy`), not these full names, and multiple tags per
row (it's multi-label). `scripts/common.py::FRAME_TAG_ALIASES` maps observed short
tags to this canonical set; anything it can't map is flagged by `flag_issues.py`
as `unknown_frame_tag` for manual review rather than silently dropped or guessed at.

## Column reference

| Column | Represents |
|---|---|
| `uuid` | Unique article ID |
| `title` | Headline |
| `date_publish` | Publication timestamp |
| `source_domain` | Publisher domain |
| `url` | Original article link (only path to real source content) |
| `political_leaning` | Publisher bias: left / left_lean / center / right_lean / right |
| `text-topic` (+`-exp`) | LLM-derived main subject of the article, with justification |
| `text-entity-name` / `-sentiment` (+`-exp`) | Key entity in the text and sentiment toward it |
| `text-generic-frame` (+`-exp`) | **Original main label**, from the dataset. Multi-label set from the 15-category taxonomy, stringified list |
| `text-issue-frame` (+`-exp`) | Free-form, issue-specific frame (not fixed vocabulary), e.g. "Geopolitical Tension" |
| `img-generic-frame` (+`-exp`) | **Original main label**, from the dataset. Same 15-category taxonomy applied to the lead image |
| `img-entity-name` / `-sentiment` (+`-exp`) | Key visual entity/subject and sentiment conveyed |
| `gpt-topic` | Separate, broader GPT-generated topic classification |
| `new_text_generic_frame` (+`_strengths`, `_exp`) | **Single-model relabel** (claude-haiku-4.5) of the text, strong/moderate-only, with per-frame strength and an explanation — see `data/sample_relabeled.jsonl` / `report_overlap.py` |
| `new_img_generic_frame` (+`_strengths`, `_exp`) | **Single-model relabel** (claude-haiku-4.5) of the image, strong/moderate-only (empty = no frame applies), with per-frame strength and an explanation |
| `by_model` (in `sample_consolidated.jsonl`) | Each of the 3 ensemble models' individual text/image frame sets, strengths, and explanations |
| `consolidated_text_generic_frame` / `consolidated_img_generic_frame` (+`_votes`) | **This project's main label**: 2-of-3 majority vote across the ensemble, with per-frame vote counts |

## What this project adds on top

- `scripts/build_sample.py` — draws from a seeded shuffle of the dataset and
  scrapes each row's original article text (`trafilatura`) and lead image
  (`og:image`/`twitter:image`), continuing until 300 rows have **both**
  successfully — because a one-shot random 300 loses roughly half its rows to
  link rot/paywalls on these 2023-2024 articles. This trades strict random-
  sampling purity for guaranteed scrapeability: the final sample is biased
  toward outlets/links still live and unpaywalled. Every attempt (kept or
  rejected, and why) is logged to `data/scrape_attempts_log.jsonl`.
- `scripts/inspect_columns.py` — column-by-column stats + observed frame-tag
  frequency for the sample, cross-checked against the taxonomy above.
- `scripts/filter_news.py` — drops rows that aren't genuine news/journalism
  (game/entertainment announcements, "best of" listicles, lifestyle content) —
  framing analysis assumes editorial choices about a real-world issue, which
  doesn't apply to that kind of content. Uses a cheap text-only LLM call per row.
  Removed 62/300 rows, leaving 238 (no replenishment).
- `scripts/relabel_frames.py` — **ignores the dataset's original labels** and
  generates fresh ones, run once per model in `common.ENSEMBLE_MODELS`
  (`anthropic/claude-haiku-4.5`, `openai/gpt-5.4-mini`, `google/gemini-3.5-flash`
  — one from each of three distinct training pipelines, at a comparable
  cheap/fast cost tier for a fair comparison): two independent LLM calls per
  row (text first, then image — the image call never sees the text call's
  output, so any divergence is genuine signal), each returning only frames
  that apply **strongly or moderately** (weak/tangential connections are
  dropped entirely, not just hidden), with per-frame strength and an
  explanation. An empty image frame set is an explicitly valid "no framing"
  outcome — most news images are purely illustrative (e.g. a plain storefront
  photo in a story about that store). Taxonomy definitions passed to the model
  are the codebook's exact wording (`common.py::CANONICAL_FRAMES`).
- `scripts/report_overlap.py` — compares the single-model (claude-haiku-4.5)
  relabeling against the original dataset labels (Jaccard overlap,
  most-added/dropped frames) and reports how often the new image frame set is
  a subset of the new text frame set — useful signal when it isn't.
- `scripts/consolidate_annotations.py` — combines the 3 models' independent
  label sets into one: a frame is kept only if at least 2 of 3 models
  independently assigned it, for text and image separately. Each model's raw
  output is kept alongside (`by_model`) for transparency. Also reports
  pairwise agreement between models and how often all 3 agreed exactly.
- `scripts/flag_issues.py` — flags rows worth a closer look: structural issues
  in the original columns, a model's relabel call failing, low ensemble
  agreement, consolidated image frames not a subset of text frames, and large
  disagreement between the consolidated label and the original.
- `app/validate_app.py` — Streamlit UI for the manual validation pass: shows
  the article text + image, the original labels, the 2-of-3 consolidated
  labels, and each model's individual labels (expandable per modality),
  defaulting to showing all 238 rows (toggle to flagged-only).
- `scripts/baseline_qwen_text.py` / `baseline_qwen_vlm.py` / `report_baselines.py`
  — small open-weight model baselines, run locally on GPU rather than via
  OpenRouter: `Qwen/Qwen3-4B-Instruct-2507` zero-shot text framing, and
  `Qwen/Qwen3-VL-4B-Instruct` zero-shot image framing in two settings (image
  alone vs. image + the ground-truth text frame as context), both compared
  against the consolidated ensemble labels. Finding: giving the VLM the
  ground-truth text frame made image-frame agreement *worse* (0.43 vs. 0.52
  Jaccard blind), with 54% of oracle-setting predictions being an exact copy
  of the given text label rather than an independent visual judgment.
