# Consolidation report

Models: anthropic/claude-haiku-4.5, openai/gpt-5.4-mini, google/gemini-3.5-flash
Agreement threshold: 2 of 3
Rows: 238

### Text frames
- Avg pairwise Jaccard across model pairs: 0.60
- Rows where all 3 models produced identical sets: 15/238 (6%)
- Avg frames kept per row after 2-of-3 consolidation: 2.44
- Rows with zero frames surviving consolidation: 1/238

### Image frames
- Avg pairwise Jaccard across model pairs: 0.49
- Rows where all 3 models produced identical sets: 36/238 (15%)
- Avg frames kept per row after 2-of-3 consolidation: 1.40
- Rows with zero frames surviving consolidation ("None"): 30/238

### Model call errors
- google_gemini-3_5-flash_text: 1
- google_gemini-3_5-flash_img: 1