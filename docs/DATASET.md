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
  the whole set. Treat every row as an **LLM prediction**, not ground truth.

## Splits

- `full` (`annotated_data.csv`, ~479K rows) — everything.
- `valid_framing_subset` (`framing_subset.csv`, ~154K rows) — the paper's own
  filtered, framing-analysis-ready subset: drops "None"/invalid frame predictions,
  articles under 100 words, and Sports/Media topics. **This is what
  `scripts/sample_dataset.py` samples from by default.**

## Important: the dataset stores labels, not source content

Neither the raw article text nor the image (or even an image URL) is stored as a
column. The only columns are metadata (`title`, `date_publish`, `source_domain`,
`political_leaning`, `url`) and the **LLM's output about** the text/image (labels +
free-text justifications). The only path back to the actual source content is
`url` — hence `scripts/scrape_articles.py`, which is a best-effort live fetch and
will have real link-rot for two-year-old articles.

## The 15-category generic frame taxonomy

Both `text-generic-frame` and `img-generic-frame` — **the two main labels this
project cares about** — draw from the same fixed, multi-label taxonomy, adapted
from Boydstun et al. (2014) / the Media Frames Corpus:

| Frame | Definition |
|---|---|
| Economic | Costs, benefits, or monetary/financial implications |
| Capacity & Resources | Availability (or lack) of physical, geographic, spatial, human, financial resources |
| Morality | Perspective compelled by religious doctrine, duty, honor, righteousness |
| Fairness & Equality | Equality/inequality in how laws, punishment, rewards, resources are applied |
| Legality, Constitutionality & Jurisprudence | Constraints/freedoms via Constitution and judicial interpretation |
| Policy Prescription & Evaluation | Specific policies proposed to address a problem |
| Crime & Punishment | Enforcement/interpretation of laws, lawbreaking, sentencing |
| Security & Defense | Security, threats to it, protection of person/family/nation |
| Health & Safety | Healthcare access, illness, disease, sanitation, violence prevention |
| Quality of Life | Effects on wealth, mobility, access to resources, happiness |
| Cultural Identity | Social norms, trends, values, customs |
| Public Opinion | General social attitudes, polling, demographics |
| Political | Partisan maneuvering, lobbying, bipartisan deal-making |
| External Regulation & Reputation | A country's external relations, trade agreements |
| Other | Doesn't fit the above |

**Caveat:** the raw CSV stores short-form tags (e.g. `security`, `legality`,
`regulation`, `policy`), not these full names, and multiple tags per row (it's
multi-label). `scripts/common.py::FRAME_TAG_ALIASES` maps observed short tags to
this canonical set; anything it can't map is flagged by `flag_issues.py` as
`unknown_frame_tag` for manual review rather than silently dropped or guessed at.

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
| `text-generic-frame` (+`-exp`) | **Main label.** Multi-label set from the 15-category taxonomy, stringified list |
| `text-issue-frame` (+`-exp`) | Free-form, issue-specific frame (not fixed vocabulary), e.g. "Geopolitical Tension" |
| `img-generic-frame` (+`-exp`) | **Main label.** Same 15-category taxonomy applied to the lead image |
| `img-entity-name` / `-sentiment` (+`-exp`) | Key visual entity/subject and sentiment conveyed |
| `gpt-topic` | Separate, broader GPT-generated topic classification |

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
- `scripts/judge_frames.py` — the semantic check: shows a vision LLM
  (`anthropic/claude-haiku-4.5` via OpenRouter) the real scraped article text
  and image for each row, plus the `text-generic-frame`/`img-generic-frame`
  labels and the original labeling model's own justification, and asks it to
  independently judge whether each label actually holds up — not just whether
  the justification sounds plausible.
- `scripts/flag_issues.py` — merges those semantic verdicts with cheap
  structural/statistical checks (missing fields, malformed list literals,
  out-of-taxonomy tags, suspiciously thin LLM explanations, out-of-range
  dates) into one flags file, so the human reviewer is pointed at what's worth
  their time.
- `app/validate_app.py` — Streamlit UI for the manual validation pass: shows
  the article text + image + both frame labels + the automated judge's
  verdict and reasoning per row, defaulting to flagged rows first.
