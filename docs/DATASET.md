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
  the whole set. Every row is an **LLM prediction**, not ground truth — this
  project's own labeling pipeline (see the main [README](../README.md)) treats it
  as exactly that.

## Splits

- `full` (`annotated_data.csv`, ~479K rows) — everything.
- `valid_framing_subset` (`framing_subset.csv`, ~154K rows) — the paper's own
  filtered, framing-analysis-ready subset: drops "None"/invalid frame predictions,
  articles under 100 words, and Sports/Media topics. This is the split this
  project samples from.

## The dataset stores labels, not source content

Neither the raw article text nor the image (or even an image URL) is stored as a
column. The only columns are metadata (`title`, `date_publish`, `source_domain`,
`political_leaning`, `url`) and the **LLM's output about** the text/image (labels +
free-text justifications). The only path back to the actual source content is
`url`, which is why this project scrapes each sampled article live.

## The 15-category generic frame taxonomy

Both `text-generic-frame` and `img-generic-frame` draw from the same fixed,
multi-label taxonomy, adapted from Boydstun et al. (2014) / the Media Frames
Corpus. Definitions below are the codebook's own wording, verbatim:

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
tags to this canonical set.

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
| `text-generic-frame` (+`-exp`) | Original text frame label from the dataset — multi-label, 15-category taxonomy, stringified list |
| `text-issue-frame` (+`-exp`) | Free-form, issue-specific frame (not fixed vocabulary), e.g. "Geopolitical Tension" |
| `img-generic-frame` (+`-exp`) | Original image frame label from the dataset — same 15-category taxonomy applied to the lead image |
| `img-entity-name` / `-sentiment` (+`-exp`) | Key visual entity/subject and sentiment conveyed |
| `gpt-topic` | Separate, broader GPT-generated topic classification |

This project's own labels (`consolidated_text_generic_frame`,
`consolidated_img_generic_frame`, per-model breakdowns, and the baseline
predictions) are described in the main [README](../README.md) and
[`docs/PIPELINE.md`](PIPELINE.md).
