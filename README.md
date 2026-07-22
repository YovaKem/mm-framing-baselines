# mm-framing-baselines

Baselines for multimodal news-framing classification, built on
[`copenlu/mm-framing`](https://huggingface.co/datasets/copenlu/mm-framing) — the
dataset behind [*Multi-Modal Framing Analysis of News*](https://arxiv.org/abs/2503.20960)
(Arora et al.), which studies how news articles frame their subject through both
text and accompanying photographs, using a 15-category taxonomy
([Boydstun et al.](https://mfc.osu.edu/)).

## Method

**Sample.** 600 articles come from the paper's own small human
double-annotation pass over images (see [`docs/DATASET.md`](docs/DATASET.md)) —
real human ground truth for image frames, not an LLM proxy. 578 of those 600
uuids have a matching row (url, title, metadata) in the full `copenlu/mm-framing`
dataset (22 have no match anywhere and are dropped). Of those 578, only
**~400 are actually scrapeable** — CBS News alone accounts for 264/578 (46%) of
this sample, and CBS's WAF blocks both live scraping and, largely, the Wayback
Machine's own crawler too (see the sampling caveat in
[`docs/PIPELINE.md`](docs/PIPELINE.md)). The
**400 articles** are split into **300 test** (the only rows any baseline is
ever scored against) and **100 train_lora** (training data for the two LoRA
finetunes below, never touched by evaluation).

**Taxonomy.** 14 generic frames + an explicit **"None"** option, matching the
paper's own text- and image-framing prompts exactly (arXiv:2503.20960, PDF
pp. 18–19).

**Target labels.** **Image frames** are the human double-annotation labels
(`data/human_image_frame_labels.csv`). **Text frames** are
**silver standard** (the paper does not provide human labels for these articles:
three LLMs (`anthropic/claude-haiku-4.5`,
`openai/gpt-5.4-mini`, `google/gemini-3.5-flash`) each independently label
every article's text frame(s) zero-shot, from the same taxonomy, and a frame
becomes the target if at least 2 of the 3 independently assign it. The three
labeling LLMs agree with each other at a moderate level — avg. pairwise
overlap 0.57 (see [`data/consolidation_report.md`](data/consolidation_report.md))
— similar to the human agreement reported in the paper.
Across all 400 articles, **18% of images carry no frame at all**
("None") vs. 11/400 for text.

**Baselines.** Six baselines, all scored on the same 300 test rows:
- **Text** — `Qwen3-4B-Instruct` predicts the text frame from the article
  alone, zero-shot and LoRA-finetuned (trained on the 100 train_lora rows,
  target = the ensemble consensus above).
- **Image** — `Qwen3-VL-4B-Instruct` predicts the image frame, zero-shot and
  LoRA-finetuned, each in two settings: *with oracle* (also given the
  consensus text frame as context) and *without* (image + article only).
  Target = the human annotations. 

All prompts (text and image) follow the paper's own text- and
image-framing prompts verbatim, with two deliberate deviations: the
issue-specific-frame sub-task is dropped (out of scope here), and the image
prompt is given the full article text as context even though the paper's own
image prompt is image-only.

## Results

Subtask 1

| Baseline | Precision | Recall | F1 | Non-zero intersection |
|---|---:|---:|---:|---:|
| Text, zero-shot | 0.53 | 0.61 | 0.57 | 92% |
| Text, LoRA-finetuned | 0.75 | 0.75 | **0.75** | 99% |

Subtask 2a (no oracle text frames)

| Baseline | Precision | Recall | F1 | Non-zero intersection |
|---|---:|---:|---:|---:|
| Image, zero-shot | 0.40 | 0.36 | 0.38 | 59% |
| Image, LoRA-finetuned | 0.48 | 0.43 | **0.45** | 76% |

Subtask 2b (with oracle text frames)

| Baseline | Precision | Recall | F1 | Non-zero intersection |
|---|---:|---:|---:|---:|
| Image, zero-shot | 0.30 | 0.42 | 0.35 | 65% |
| Image, LoRA-finetuned | 0.43 | 0.41 | 0.42 | 67% |

(Metrics follow the original paper's own evaluation methodology — micro-averaged
precision/recall/F1, plus the paper's own headline **non-zero-intersection rate**
(they reported 95.7% for text, 84.2% for image): the share of rows where the
predicted and gold frame sets share at least one label. "None" is a real,
selectable label in this taxonomy (annotators and models can both assign it),
but the paper's own NZI figure for text was benchmarked against the Media
Frames Corpus (Card et al. 2015) — a separate, pre-existing dataset whose own
annotation task has no "None" option, so that particular gold set is never
empty. We compute NZI the same way here — only over rows with a non-empty
gold label — for direct comparability; rows where gold genuinely is empty get
their own separate "None-agreement" stat instead. Full per-baseline
breakdown is in [`data/baselines_report.md`](data/baselines_report.md).)

Three things stand out:

1. **LoRA finetuning helps across the board** — even with only 100 training
   examples, both text (+0.18 F1) and image (+0.07 F1 in both oracle settings)
   improve over their zero-shot counterparts.
2. **The oracle text frames hurt image F1, in both zero-shot and finetuned settings**
   — we cannot be sure if this is a real finding or a consequence of the silver-standard
   text framing labels.
4. **Finetuning makes the model far less reliant on the oracle text frames.** Of rows with
   a non-empty oracle text frame, the zero-shot oracle setting's image
   prediction is an *exact copy* of the given text frame **59%** of the time —
   after LoRA finetuning, that drops to **9%**. The finetuned model forms much
   more of its own, image-grounded judgment instead of parroting the hint,
   which tracks with its F1 no longer dropping as sharply when given the
   (unhelpful) oracle context. Full breakdown in
   [`data/baselines_report.md`](data/baselines_report.md).

## Examples

[`examples/`](examples/) has a few sampled articles shown end-to-end — the
original image and text alongside the final target labels — including one
where the image's frame diverges from the text's, and one where the image has
no frame at all. These predate the human-ground-truth sample above (drawn
from the project's earlier 238-article LLM-only pass) and haven't been
regenerated against it yet.

## Repository

```
scripts/    the pipeline: sampling, LLM text-labeling, consensus, baselines, LoRA finetunes
examples/   a few annotated articles shown end-to-end (image + text + labels)
docs/       dataset schema/taxonomy reference, and how to reproduce each step
data/       results and reports (raw scraped content is git-ignored)
```

See [`docs/DATASET.md`](docs/DATASET.md) for the dataset's schema and taxonomy,
and [`docs/PIPELINE.md`](docs/PIPELINE.md) for how to run or reproduce any step.
