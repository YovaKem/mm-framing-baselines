# Relabeling overlap report

Rows compared: 238

### text-generic-frame
- Average Jaccard overlap (old vs new): **0.46**
- Identical label sets: 15/238 (6%)
- Completely disjoint (no shared labels, at least one non-empty): 5/238 (2%)
- Average labels per row — old: 3.7, new: 2.3

### img-generic-frame
- Average Jaccard overlap (old vs new): **0.36**
- Identical label sets: 35/238 (15%)
- Completely disjoint (no shared labels, at least one non-empty): 86/238 (36%)
- Average labels per row — old: 1.3, new: 1.7

### Most frequently dropped from text (present in old, absent in new)
- Policy Prescription & Evaluation: 66
- Legality, Constitutionality & Jurisprudence: 59
- Public Opinion: 51
- Crime & Punishment: 50
- Fairness & Equality: 45
- Quality of Life: 36
- Economic: 32
- Political: 24
- External Regulation & Reputation: 23
- Cultural Identity: 18

### Most frequently added to text (absent in old, present in new)
- Political: 21
- Capacity & Resources: 20
- Security & Defense: 17
- Health & Safety: 15
- Quality of Life: 13
- Public Opinion: 12
- Legality, Constitutionality & Jurisprudence: 7
- Policy Prescription & Evaluation: 5
- Crime & Punishment: 4
- Economic: 4

### Most frequently dropped from image (present in old, absent in new)
- Public Opinion: 24
- Legality, Constitutionality & Jurisprudence: 22
- Cultural Identity: 20
- Policy Prescription & Evaluation: 17
- Capacity & Resources: 12
- Fairness & Equality: 10
- Crime & Punishment: 8
- Security & Defense: 8
- Quality of Life: 8
- Health & Safety: 8

### Most frequently added to image (absent in old, present in new)
- Political: 49
- Security & Defense: 33
- Crime & Punishment: 30
- Economic: 22
- Quality of Life: 22
- Health & Safety: 21
- Legality, Constitutionality & Jurisprudence: 16
- External Regulation & Reputation: 10
- Capacity & Resources: 9
- Public Opinion: 8

### Image-is-subset-of-text check (NEW labels only)
- New image frame set is empty ("None"): 13/238 (5%) — vs. 0/238 (0%) under the old labels
- Of the 225 rows with a non-empty new image frame set, 145 (64%) are a subset of the new text frame set — the remaining 80 have the image conveying a frame the text doesn't.