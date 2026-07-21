# Baseline report

Rows: 238

Metrics follow the original paper's methodology: micro-averaged precision/recall/F1 (computed across all row x frame pairs) and a non-zero-intersection rate. The paper's data never had an empty ("None") gold label, so non-zero-intersection is computed only over rows with a non-empty gold label here too; rows where gold genuinely is empty get their own "None-agreement" stat instead.

## Summary

| Baseline | Precision | Recall | F1 | Non-zero intersection | Identical | Avg labels (gold) | Avg labels (pred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 0.58 | 0.60 | 0.59 | 221/237 (93%) | 27/238 (11%) | 2.44 | 2.53 |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.50 | 0.56 | 0.53 | 144/206 (70%) | 62/238 (26%) | 1.45 | 1.63 |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.54 | 0.71 | 0.61 | 170/206 (83%) | 62/238 (26%) | 1.45 | 1.89 |

### "None" (empty gold label) agreement

| Baseline | Rows where gold is None | Baseline also predicted None |
|---|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 1/238 | 0/1 (0%) |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 32/238 | 20/32 (62%) |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 32/238 | 25/32 (78%) |

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
| Political | 37 | | Policy Prescription & Evaluation | 35 |
| Capacity & Resources | 17 | | Legality, Constitutionality & Jurisprudence | 27 |
| Security & Defense | 16 | | Crime & Punishment | 22 |
| Health & Safety | 12 | | Public Opinion | 19 |
| Crime & Punishment | 12 | | Health & Safety | 17 |
| Quality of Life | 12 | | Quality of Life | 15 |
| Public Opinion | 12 | | Security & Defense | 11 |
| Economic | 9 | | Fairness & Equality | 11 |

## Image, with oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Political | 18 | | Legality, Constitutionality & Jurisprudence | 33 |
| Quality of Life | 13 | | Policy Prescription & Evaluation | 30 |
| Security & Defense | 12 | | Crime & Punishment | 23 |
| Health & Safety | 11 | | Health & Safety | 20 |
| Capacity & Resources | 10 | | Political | 18 |
| Public Opinion | 9 | | Economic | 18 |
| Economic | 7 | | Capacity & Resources | 15 |
| Cultural Identity | 6 | | Quality of Life | 15 |

## Does the oracle text frame help image prediction, or just get copied?

| | No oracle | With oracle |
|---|---:|---:|
| Precision | 0.50 | 0.54 |
| Recall | 0.56 | 0.71 |
| F1 | 0.53 | 0.61 |

| Outcome of adding the oracle (per-row F1 change) | Rows |
|---|---:|
| Improved agreement | 94/238 (39%) |
| Worsened agreement | 66/238 (28%) |
| Unchanged | 78/238 (33%) |

Of 237 rows with a non-empty oracle text frame, the oracle-setting image prediction was an **exact copy** of the text frame in **169 (71%)** — a high rate here suggests the model leans on the given text label rather than looking at the image.