# Baseline report

Rows: 238

## Summary

| Baseline | Avg Jaccard | Identical | Disjoint | Avg labels (gold) | Avg labels (pred) |
|---|---:|---:|---:|---:|---:|
| Text — Qwen/Qwen3-4B-Instruct-2507 | 0.47 | 27/238 (11%) | 17/238 (7%) | 2.44 | 2.53 |
| Image, no oracle — Qwen/Qwen3-VL-4B-Instruct | 0.52 | 77/238 (32%) | 61/238 (26%) | 1.45 | 1.19 |
| Image, with oracle — Qwen/Qwen3-VL-4B-Instruct | 0.43 | 58/238 (24%) | 78/238 (33%) | 1.45 | 1.42 |

## Text — Qwen/Qwen3-4B-Instruct-2507

**Most-missed** (in ground truth, baseline didn't predict) vs. **most over-predicted** (baseline predicted, not in ground truth):

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

**Most-missed** (in ground truth, baseline didn't predict) vs. **most over-predicted** (baseline predicted, not in ground truth):

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

**Most-missed** (in ground truth, baseline didn't predict) vs. **most over-predicted** (baseline predicted, not in ground truth):

| Missed | Rows | | Over-predicted | Rows |
|---|---:|---|---|---:|
| Political | 38 | | Legality, Constitutionality & Jurisprudence | 24 |
| Security & Defense | 16 | | Policy Prescription & Evaluation | 21 |
| Health & Safety | 15 | | Health & Safety | 15 |
| Quality of Life | 15 | | Crime & Punishment | 15 |
| Public Opinion | 14 | | Political | 13 |
| Capacity & Resources | 12 | | Economic | 11 |
| Crime & Punishment | 10 | | Capacity & Resources | 11 |
| Economic | 9 | | Quality of Life | 9 |

## Does the oracle text frame help image prediction, or just get copied?

| | No oracle | With oracle |
|---|---:|---:|
| Avg Jaccard vs. ground truth | 0.52 | 0.43 |

| Outcome of adding the oracle | Rows |
|---|---:|
| Improved agreement | 63/238 (26%) |
| Worsened agreement | 89/238 (37%) |
| Unchanged | 86/238 (36%) |

Of 237 rows with a non-empty oracle text frame, the oracle-setting image prediction was an **exact copy** of the text frame in **129 (54%)** — a high rate here suggests the model leans on the given text label rather than looking at the image.