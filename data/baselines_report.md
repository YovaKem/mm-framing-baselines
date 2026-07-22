# Baseline report

Rows: 300 (split=test)

Metrics follow the original paper's methodology: micro-averaged precision/recall/F1 (computed across all row x frame pairs) and a non-zero-intersection rate. The paper's own headline NZI figure for text was benchmarked against the Media Frames Corpus (Card et al. 2015) — a separate, pre-existing human-annotated dataset whose own annotation task has no "None" option, so that particular gold set is never empty (this project's own taxonomy does include "None" as a real, selectable label). Non-zero-intersection is computed only over rows with a non-empty gold label here too, for comparability; rows where gold genuinely is empty get their own "None-agreement" stat instead.

## Summary

| Baseline | Precision | Recall | F1 | Non-zero intersection | Identical | Avg labels (gold) | Avg labels (pred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text, zero-shot — Qwen/Qwen3-4B-Instruct-2507 | 0.53 | 0.61 | 0.57 | 270/292 (92%) | 25/300 (8%) | 2.96 | 3.41 |
| Text, LoRA-finetuned — Qwen/Qwen3-4B-Instruct-2507 | 0.75 | 0.75 | 0.75 | 290/292 (99%) | 68/300 (23%) | 2.96 | 2.96 |
| Image, zero-shot, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.40 | 0.36 | 0.38 | 142/241 (59%) | 55/300 (18%) | 1.66 | 1.50 |
| Image, zero-shot, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.30 | 0.42 | 0.35 | 156/241 (65%) | 24/300 (8%) | 1.66 | 2.33 |
| Image, LoRA-finetuned, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.48 | 0.43 | 0.45 | 183/241 (76%) | 70/300 (23%) | 1.66 | 1.47 |
| Image, LoRA-finetuned, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.43 | 0.41 | 0.42 | 161/241 (67%) | 63/300 (21%) | 1.66 | 1.57 |

### "None" (empty gold label) agreement

| Baseline | Rows where gold is None | Baseline also predicted None |
|---|---:|---:|
| Text, zero-shot — Qwen/Qwen3-4B-Instruct-2507 | 8/300 | 5/8 (62%) |
| Text, LoRA-finetuned — Qwen/Qwen3-4B-Instruct-2507 | 8/300 | 1/8 (12%) |
| Image, zero-shot, no oracle — Qwen/Qwen3-VL-4B-Instruct | 59/300 | 17/59 (29%) |
| Image, zero-shot, with oracle — Qwen/Qwen3-VL-4B-Instruct | 59/300 | 10/59 (17%) |
| Image, LoRA-finetuned, no oracle — Qwen/Qwen3-VL-4B-Instruct | 59/300 | 14/59 (24%) |
| Image, LoRA-finetuned, with oracle — Qwen/Qwen3-VL-4B-Instruct | 59/300 | 19/59 (32%) |

## Text, zero-shot — Qwen/Qwen3-4B-Instruct-2507

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Capacity & Resources | 64 | | Fairness & Equality | 100 |
| Political | 55 | | Public Opinion | 98 |
| Legality, Constitutionality & Jurisprudence | 43 | | Morality | 70 |
| Health & Safety | 34 | | Quality of Life | 57 |
| Crime & Punishment | 33 | | Policy Prescription & Evaluation | 48 |
| Economic | 32 | | Cultural Identity | 30 |
| Security & Defense | 20 | | Health & Safety | 29 |
| Quality of Life | 17 | | Security & Defense | 16 |

## Text, LoRA-finetuned — Qwen/Qwen3-4B-Instruct-2507

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Economic | 33 | | Public Opinion | 40 |
| Capacity & Resources | 28 | | Quality of Life | 39 |
| Legality, Constitutionality & Jurisprudence | 27 | | Capacity & Resources | 27 |
| Fairness & Equality | 24 | | Policy Prescription & Evaluation | 26 |
| Political | 18 | | Health & Safety | 23 |
| Policy Prescription & Evaluation | 17 | | Cultural Identity | 22 |
| Quality of Life | 15 | | Crime & Punishment | 18 |
| Health & Safety | 13 | | Political | 11 |

## Image, zero-shot, no oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Cultural Identity | 48 | | Quality of Life | 58 |
| Quality of Life | 47 | | Capacity & Resources | 32 |
| Economic | 41 | | Public Opinion | 32 |
| Political | 33 | | Policy Prescription & Evaluation | 32 |
| Capacity & Resources | 28 | | Crime & Punishment | 25 |
| Health & Safety | 28 | | Legality, Constitutionality & Jurisprudence | 20 |
| Security & Defense | 22 | | Health & Safety | 13 |
| Public Opinion | 17 | | Political | 13 |

## Image, zero-shot, with oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Quality of Life | 46 | | Quality of Life | 61 |
| Cultural Identity | 46 | | Capacity & Resources | 61 |
| Economic | 30 | | Legality, Constitutionality & Jurisprudence | 60 |
| Capacity & Resources | 26 | | Health & Safety | 53 |
| Public Opinion | 26 | | Policy Prescription & Evaluation | 42 |
| Political | 21 | | Crime & Punishment | 37 |
| Health & Safety | 19 | | Economic | 36 |
| Security & Defense | 19 | | Political | 35 |

## Image, LoRA-finetuned, no oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Quality of Life | 40 | | Health & Safety | 57 |
| Economic | 31 | | Quality of Life | 42 |
| Capacity & Resources | 31 | | Cultural Identity | 29 |
| Cultural Identity | 31 | | Public Opinion | 19 |
| Political | 25 | | Political | 16 |
| Crime & Punishment | 23 | | Security & Defense | 16 |
| Public Opinion | 20 | | Crime & Punishment | 13 |
| Security & Defense | 20 | | Fairness & Equality | 11 |

## Image, LoRA-finetuned, with oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Quality of Life | 51 | | Quality of Life | 63 |
| Cultural Identity | 39 | | Health & Safety | 40 |
| Capacity & Resources | 33 | | Political | 29 |
| Economic | 31 | | Security & Defense | 25 |
| Public Opinion | 23 | | Crime & Punishment | 23 |
| Political | 20 | | Cultural Identity | 21 |
| Security & Defense | 18 | | Public Opinion | 17 |
| Crime & Punishment | 17 | | Economic | 13 |

## Does the oracle text frame help zero-shot image prediction, or just get copied?

| | No oracle | With oracle |
|---|---:|---:|
| Precision | 0.40 | 0.30 |
| Recall | 0.36 | 0.42 |
| F1 | 0.38 | 0.35 |

| Outcome of adding the oracle (per-row F1 change) | Rows |
|---|---:|
| Improved agreement | 56/300 (19%) |
| Worsened agreement | 84/300 (28%) |
| Unchanged | 160/300 (53%) |

Of 292 rows with a non-empty oracle text frame, the oracle-setting image prediction was an **exact copy** of the text frame in **173 (59%)** — a high rate here suggests the model leans on the given text label rather than looking at the image.

## Does the oracle text frame help LoRA-finetuned image prediction, or just get copied?

| | No oracle | With oracle |
|---|---:|---:|
| Precision | 0.48 | 0.43 |
| Recall | 0.43 | 0.41 |
| F1 | 0.45 | 0.42 |

| Outcome of adding the oracle (per-row F1 change) | Rows |
|---|---:|
| Improved agreement | 50/300 (17%) |
| Worsened agreement | 75/300 (25%) |
| Unchanged | 175/300 (58%) |

Of 292 rows with a non-empty oracle text frame, the oracle-setting image prediction was an **exact copy** of the text frame in **25 (9%)** — a high rate here suggests the model leans on the given text label rather than looking at the image.