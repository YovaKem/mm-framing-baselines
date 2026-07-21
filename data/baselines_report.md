# Baseline report

Rows: 238

Text model: `Qwen/Qwen3-4B-Instruct-2507` (zero-shot, local)
Image model: `Qwen/Qwen3-VL-4B-Instruct` (zero-shot, local, two settings)

### Text: Qwen/Qwen3-4B-Instruct-2507
- Avg Jaccard vs. consolidated ground truth: **0.47**
- Identical label sets: 27/238 (11%)
- Completely disjoint (no shared labels, at least one non-empty): 17/238 (7%)
- Avg labels per row — ground truth: 2.44, baseline: 2.53

Most-missed frames (in ground truth, baseline didn't predict):
  - Political: 58
  - Crime & Punishment: 35
  - Capacity & Resources: 20
  - Legality, Constitutionality & Jurisprudence: 19
  - Policy Prescription & Evaluation: 16
  - Health & Safety: 15
  - External Regulation & Reputation: 14
  - Economic: 14
Most-over-predicted frames (baseline predicted, not in ground truth):
  - Fairness & Equality: 72
  - Quality of Life: 51
  - Public Opinion: 25
  - Morality: 23
  - Policy Prescription & Evaluation: 20
  - Legality, Constitutionality & Jurisprudence: 17
  - Health & Safety: 13
  - Crime & Punishment: 10

### Image (no oracle): Qwen/Qwen3-VL-4B-Instruct
- Avg Jaccard vs. consolidated ground truth: **0.52**
- Identical label sets: 77/238 (32%)
- Completely disjoint (no shared labels, at least one non-empty): 61/238 (26%)
- Avg labels per row — ground truth: 1.45, baseline: 1.19

Most-missed frames (in ground truth, baseline didn't predict):
  - Political: 26
  - Capacity & Resources: 17
  - Quality of Life: 16
  - Crime & Punishment: 14
  - Public Opinion: 14
  - Legality, Constitutionality & Jurisprudence: 13
  - Health & Safety: 12
  - Security & Defense: 12
Most-over-predicted frames (baseline predicted, not in ground truth):
  - Policy Prescription & Evaluation: 16
  - Legality, Constitutionality & Jurisprudence: 15
  - Health & Safety: 10
  - Crime & Punishment: 10
  - Quality of Life: 9
  - Political: 7
  - Public Opinion: 6
  - Security & Defense: 6

### Image (with oracle text frame): Qwen/Qwen3-VL-4B-Instruct
- Avg Jaccard vs. consolidated ground truth: **0.43**
- Identical label sets: 58/238 (24%)
- Completely disjoint (no shared labels, at least one non-empty): 78/238 (33%)
- Avg labels per row — ground truth: 1.45, baseline: 1.42

Most-missed frames (in ground truth, baseline didn't predict):
  - Political: 38
  - Security & Defense: 16
  - Health & Safety: 15
  - Quality of Life: 15
  - Public Opinion: 14
  - Capacity & Resources: 12
  - Crime & Punishment: 10
  - Economic: 9
Most-over-predicted frames (baseline predicted, not in ground truth):
  - Legality, Constitutionality & Jurisprudence: 24
  - Policy Prescription & Evaluation: 21
  - Health & Safety: 15
  - Crime & Punishment: 15
  - Political: 13
  - Economic: 11
  - Capacity & Resources: 11
  - Quality of Life: 9

### Does the oracle text frame help image prediction, or just get copied?
- Avg Jaccard vs. ground truth — no oracle: 0.52, with oracle: 0.43
- Rows where oracle setting improved agreement: 63/238 (26%)
- Rows where oracle setting worsened agreement: 89/238 (37%)
- Rows unchanged: 86/238 (36%)
- Of 237 rows with a non-empty oracle text frame, the oracle-setting image prediction was an EXACT copy of the text frame in 129 (54%) — a high rate here would suggest the model leans on the given text label rather than looking at the image.