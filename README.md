# mm-framing-baselines

Baselines for multimodal news-framing classification, built on
[`copenlu/mm-framing`](https://huggingface.co/datasets/copenlu/mm-framing) — the
dataset behind [*Multi-Modal Framing Analysis of News*](https://arxiv.org/abs/2503.20960)
(Arora et al.), which studies how news articles frame their subject through both
text and accompanying photographs, using a 15-category taxonomy
([Boydstun et al.](https://mfc.osu.edu/)).

## Method

**Sample.** 300 articles were drawn at random from the dataset's
`valid_framing_subset` split, with each article's original text and lead image
scraped live from its source URL (the dataset itself stores only derived labels,
not the source content — see [`docs/DATASET.md`](docs/DATASET.md)).

**Filtering.** Rows that weren't genuine news/journalism — game announcements,
shopping listicles, lifestyle content — were dropped, since framing analysis only
makes sense for editorial coverage of a real-world issue. **238 articles**
remained.

**Target labels.** Three LLMs (`anthropic/claude-haiku-4.5`, `openai/gpt-5.4-mini`,
`google/gemini-3.5-flash`) each independently labeled every article's **text
frame(s)** and **image frame(s)** zero-shot, from the same 15-category taxonomy.
For every frame it assigned, a model also rated how strongly it applied —
**strong** (one of the main angles) or **moderate** (clearly present, but
secondary); anything weaker than that was dropped entirely rather than reported,
so the frame lists aren't padded with tangential matches. A frame became the
**consensus target label** if at least 2 of the 3 models independently assigned
it, at either strength — separately for text and for image. A stricter
**strong-only target** keeps a frame only if at least 2 of the 3 models
specifically rated it "strong". The three labeling LLMs agree with *each other*
at a similar level throughout — avg. pairwise overlap 0.60 for text frames,
0.49 for image frames (see [`data/consolidation_report.md`](data/consolidation_report.md))
— so this is a genuinely hard, high-disagreement labeling task even among
frontier models. After consensus, articles carry **2.44 text frames** on
average versus **1.40 image frames**; **30 of 238 images (13%)** end up with
*no* frame at all ("None"), against just 1 article with no text frame.

**Baselines.** Two small open-weight models were evaluated zero-shot against that
consensus target:
- **Text** — `Qwen3-4B-Instruct` predicts the text frame from the article alone.
- **Image** — `Qwen3-VL-4B-Instruct` predicts the image frame in two settings:
  *with oracle* (also given the consensus text frame as context) and *without*
  (image + article only, no hint about the text labeling).

## Results

**All frames (strong + moderate)**

| Baseline | Precision | Recall | F1 |
|---|---:|---:|---:|
| Text | 0.58 | 0.60 | 0.59 |
| Image, without oracle | 0.50 | 0.58 | 0.53 |
| Image, with oracle | 0.54 | 0.73 | **0.62** |

**Strong frames only** (both the target and the baseline's own prediction
restricted to frames rated "strong")

| Baseline | Precision | Recall | F1 |
|---|---:|---:|---:|
| Text | 0.63 | 0.50 | 0.56 |
| Image, without oracle | 0.48 | 0.66 | **0.55** |
| Image, with oracle | 0.33 | 0.89 | 0.48 |

(Metrics follow the original paper's own evaluation methodology — micro-averaged
precision/recall/F1 — for direct comparability.)

Giving the image model the correct text frame as context measurably helps when
moderate frames are counted too (F1 0.62 vs. 0.53) — but that **reverses** once
restricted to strong frames only (F1 0.48 vs. 0.55): precision collapses to 0.33
as the model over-applies the given text frame rather than judging the image on
its own, with recall alone climbing to 0.89. Consistent with that, 71% of
"with oracle" predictions turn out to be an exact copy of the given text frame —
a good chunk of the apparent gain is the model leaning on the handed-to-it
answer, not an independent visual judgment. Full breakdown in
[`data/baselines_report.md`](data/baselines_report.md).

## Examples

[`examples/`](examples/) has a few sampled articles shown end-to-end — the
original image and text alongside the final target labels — including one
where the image's frame diverges from the text's, and one where the image has
no frame at all.

## Repository

```
scripts/    the pipeline: sampling, filtering, LLM labeling, consensus, baselines
examples/   a few annotated articles shown end-to-end (image + text + labels)
docs/       dataset schema/taxonomy reference, and how to reproduce each step
data/       results and reports (raw scraped content is git-ignored)
```

See [`docs/DATASET.md`](docs/DATASET.md) for the dataset's schema and taxonomy,
and [`docs/PIPELINE.md`](docs/PIPELINE.md) for how to run or reproduce any step.
