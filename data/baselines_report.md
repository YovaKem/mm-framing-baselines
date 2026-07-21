# Baseline report

Rows: 238

Metrics follow the original paper's methodology: micro-averaged precision/recall/F1 (computed across all row x frame pairs) and a non-zero-intersection rate. The paper's data never had an empty ("None") gold label, so non-zero-intersection is computed only over rows with a non-empty gold label here too; rows where gold genuinely is empty get their own "None-agreement" stat instead.

## Summary

| Baseline | Precision | Recall | F1 | Non-zero intersection | Identical | Avg labels (gold) | Avg labels (pred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 0.58 | 0.60 | 0.59 | 221/237 (93%) | 27/238 (11%) | 2.44 | 2.53 |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.66 | 0.54 | 0.60 | 153/206 (74%) | 77/238 (32%) | 1.45 | 1.19 |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.57 | 0.56 | 0.56 | 133/206 (65%) | 58/238 (24%) | 1.45 | 1.42 |

### "None" (empty gold label) agreement

| Baseline | Rows where gold is None | Baseline also predicted None |
|---|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 1/238 | 0/1 (0%) |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 32/238 | 24/32 (75%) |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 32/238 | 27/32 (84%) |

## Text — Qwen/Qwen3-4B-Instruct-2507

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Political | 58 | | Fairness & Equality | 72 |
| Crime & Punishment | 35 | | Quality of Life | 51 |
| Capacity & Resources | 20 | | Public Opinion | 25 |
| Legality, Constitutionality & Jurisprudence | 19 | | Morality | 23 |
| Policy Prescription & Evaluation | 16 | | Policy Prescription & Evaluation | 20 |
| Health & Safety | 15 | | Legality, Constitutionality & Jurisprudence | 17 |
| External Regulation & Reputation | 14 | | Health & Safety | 13 |
| Economic | 14 | | Crime & Punishment | 10 |

## Image, no oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Political | 26 | | Policy Prescription & Evaluation | 16 |
| Capacity & Resources | 17 | | Legality, Constitutionality & Jurisprudence | 15 |
| Quality of Life | 16 | | Health & Safety | 10 |
| Crime & Punishment | 14 | | Crime & Punishment | 10 |
| Public Opinion | 14 | | Quality of Life | 9 |
| Legality, Constitutionality & Jurisprudence | 13 | | Political | 7 |
| Health & Safety | 12 | | Public Opinion | 6 |
| Security & Defense | 12 | | Security & Defense | 6 |

## Image, with oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Political | 38 | | Legality, Constitutionality & Jurisprudence | 24 |
| Security & Defense | 16 | | Policy Prescription & Evaluation | 21 |
| Health & Safety | 15 | | Health & Safety | 15 |
| Quality of Life | 15 | | Crime & Punishment | 15 |
| Public Opinion | 14 | | Political | 13 |
| Capacity & Resources | 12 | | Capacity & Resources | 11 |
| Crime & Punishment | 10 | | Economic | 11 |
| Economic | 9 | | Quality of Life | 9 |

## Does the oracle text frame help image prediction, or just get copied?

| | No oracle | With oracle |
|---|---:|---:|
| Precision | 0.66 | 0.57 |
| Recall | 0.54 | 0.56 |
| F1 | 0.60 | 0.56 |

| Outcome of adding the oracle (per-row F1 change) | Rows |
|---|---:|
| Improved agreement | 63/238 (26%) |
| Worsened agreement | 89/238 (37%) |
| Unchanged | 86/238 (36%) |

Of 237 rows with a non-empty oracle text frame, the oracle-setting image prediction was an **exact copy** of the text frame in **129 (54%)** — a high rate here suggests the model leans on the given text label rather than looking at the image.