# Consolidation report

Text ensemble models: anthropic/claude-haiku-4.5, openai/gpt-5.4-mini, google/gemini-3.5-flash
Agreement threshold: 2 of 3
Rows: 400 (split=test: 300, split=train_lora: 100)

### Text frames (silver standard: 2-of-3 LLM ensemble consensus)
- Avg pairwise Jaccard across model pairs: 0.57
- Rows where all 3 models produced identical sets: 19/400 (5%)
- Avg frames kept per row after 2-of-3 consolidation: 3.01
- Rows with zero frames surviving consolidation: 11/400

### Image frames (ground truth: paper's own human double-annotation)
- Avg frames per row: 1.65
- Rows with zero frames ("None"): 72/400 (18%)
- Frame frequency: Quality of Life 113, Political 85, Economic 78, Cultural Identity 78, Health & Safety 72, Public Opinion 45, Capacity & Resources 44, Security & Defense 38, Crime & Punishment 34, Fairness & Equality 23, Policy Prescription & Evaluation 20, Morality 17, External Regulation & Reputation 12