# Baseline report

Rows: 238

Metrics follow the original paper's methodology: micro-averaged precision/recall/F1 (computed across all row x frame pairs) and a non-zero-intersection rate. The paper's data never had an empty ("None") gold label, so non-zero-intersection is computed only over rows with a non-empty gold label here too; rows where gold genuinely is empty get their own "None-agreement" stat instead.

## Summary

| Baseline | Precision | Recall | F1 | Non-zero intersection | Identical | Avg labels (gold) | Avg labels (pred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 0.58 | 0.60 | 0.59 | 221/237 (93%) | 27/238 (11%) | 2.44 | 2.53 |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.50 | 0.58 | 0.53 | 143/208 (69%) | 66/238 (28%) | 1.40 | 1.63 |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.54 | 0.73 | 0.62 | 171/208 (82%) | 62/238 (26%) | 1.40 | 1.89 |

### "None" (empty gold label) agreement

| Baseline | Rows where gold is None | Baseline also predicted None |
|---|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 1/238 | 0/1 (0%) |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 30/238 | 18/30 (60%) |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 30/238 | 23/30 (77%) |

## Summary — strong framings only

Same comparison, restricted to high-confidence frames on both sides: a gold frame counts only if >=2 of the 3 ensemble models rated it "strong" (not "moderate"), and a baseline prediction counts only where the baseline itself rated that frame "strong".

| Baseline | Precision | Recall | F1 | Non-zero intersection | Identical | Avg labels (gold) | Avg labels (pred) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 0.63 | 0.50 | 0.56 | 145/221 (66%) | 98/238 (41%) | 1.23 | 0.97 |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.48 | 0.66 | 0.55 | 101/147 (69%) | 112/238 (47%) | 0.67 | 0.92 |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.33 | 0.89 | 0.48 | 131/147 (89%) | 58/238 (24%) | 0.67 | 1.82 |

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
| Political | 33 | | Policy Prescription & Evaluation | 34 |
| Security & Defense | 19 | | Legality, Constitutionality & Jurisprudence | 27 |
| Capacity & Resources | 14 | | Crime & Punishment | 24 |
| Quality of Life | 11 | | Public Opinion | 18 |
| Crime & Punishment | 10 | | Health & Safety | 16 |
| Health & Safety | 10 | | Quality of Life | 13 |
| Economic | 9 | | Security & Defense | 12 |
| Cultural Identity | 8 | | Fairness & Equality | 12 |

## Image, with oracle — Qwen/Qwen3-VL-4B-Instruct

**Most-missed** (in ground truth, baseline didn't predict — false negatives) vs. **most over-predicted** (baseline predicted, not in ground truth — false positives):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Political | 16 | | Legality, Constitutionality & Jurisprudence | 32 |
| Security & Defense | 14 | | Policy Prescription & Evaluation | 29 |
| Quality of Life | 11 | | Crime & Punishment | 26 |
| Health & Safety | 11 | | Political | 21 |
| Cultural Identity | 7 | | Health & Safety | 21 |
| Public Opinion | 7 | | Economic | 16 |
| Economic | 6 | | Capacity & Resources | 15 |
| Capacity & Resources | 6 | | Quality of Life | 12 |

## Does the oracle text frame help image prediction, or just get copied?

| | No oracle | With oracle |
|---|---:|---:|
| Precision | 0.50 | 0.54 |
| Recall | 0.58 | 0.73 |
| F1 | 0.53 | 0.62 |

| Outcome of adding the oracle (per-row F1 change) | Rows |
|---|---:|
| Improved agreement | 88/238 (37%) |
| Worsened agreement | 70/238 (29%) |
| Unchanged | 80/238 (34%) |

Of 237 rows with a non-empty oracle text frame, the oracle-setting image prediction was an **exact copy** of the text frame in **169 (71%)** — a high rate here suggests the model leans on the given text label rather than looking at the image.