# Inspection report

Source: `sample_300_scraped.jsonl`  |  Rows: 300

## Columns

| column | non-null | unique | description |
|---|---|---|---|
| `uuid` | 300/300 | 300 | Unique ID for the article. |
| `title` | 300/300 | 300 | Article headline. |
| `date_publish` | 300/300 | 300 | Publication timestamp. |
| `source_domain` | 300/300 | 24 | Publisher's domain (28 unique outlets in the full dataset). |
| `url` | 300/300 | 300 | Original article URL — the only pointer back to the source text/image; not itself the content. |
| `political_leaning` | 300/300 | 5 | Publisher bias label: left / left_lean / center / right_lean / right. |
| `text-topic` | 300/300 | 241 | Main subject of the article (LLM-derived, short phrase). |
| `text-topic-exp` | 300/300 | 300 | LLM's free-text justification for the topic label. |
| `text-entity-name` | 299/300 | 281 | Key entity/person/org the article centers on (LLM-derived). |
| `text-entity-sentiment` | 299/300 | 6 | Sentiment expressed toward that entity in the text (LLM-derived). |
| `text-entity-sentiment-exp` | 300/300 | 300 | LLM's free-text justification for the entity sentiment. |
| `text-generic-frame` | 300/300 | 233 | MAIN LABEL — multi-label set of generic frames present in the article text, drawn from the 15-category Boydstun et al. taxonomy. Stored as a stringified Python list of short tags. |
| `text-generic-frame-exp` | 300/300 | 300 | LLM's free-text justification for the text frame labels. |
| `text-issue-frame` | 300/300 | 272 | Issue-specific (non-fixed-vocabulary) framing label for the text, e.g. 'Geopolitical Tension'. |
| `text-issue-frame-exp` | 300/300 | 300 | LLM's free-text justification for the issue frame label. |
| `img-generic-frame` | 300/300 | 52 | MAIN LABEL — multi-label set of generic frames present in the article's lead image, same 15-category taxonomy as text-generic-frame. Stringified Python list of short tags. |
| `img-frame-exp` | 300/300 | 300 | LLM's free-text justification for the image frame labels. |
| `img-entity-name` | 264/300 | 215 | Key entity/subject visible in the image (LLM-derived). |
| `img-entity-sentiment` | 298/300 | 6 | Sentiment conveyed by the image toward that entity (LLM-derived). |
| `img-entity-sentiment-exp` | 298/300 | 297 | LLM's free-text justification for the image entity sentiment. |
| `gpt-topic` | 300/300 | 35 | Separate GPT-generated topic classification (broader taxonomy than text-topic). |

## political_leaning distribution

political_leaning
left_lean     135
right          65
center         58
right_lean     28
left           14


## Top 15 gpt-topic values

gpt-topic
Politics         60
Crime            51
Business         28
Health           21
Legal            19
Entertainment    18
Culture          12
War              10
Weather           8
Economy           8
Labor             7
Technology        6
Environment       6
Travel            6
Social Issues     5


## text-generic-frame: canonical tag frequency (of 15 taxonomy categories)

- Policy Prescription & Evaluation: 123
- Quality of Life: 117
- Legality, Constitutionality & Jurisprudence: 111
- Crime & Punishment: 108
- Economic: 105
- Political: 94
- Public Opinion: 78
- Fairness & Equality: 74
- Security & Defense: 69
- Health & Safety: 68
- Cultural Identity: 61
- External Regulation & Reputation: 41
- Capacity & Resources: 30
- Morality: 16

Never observed in this sample: other


## img-generic-frame: canonical tag frequency (of 15 taxonomy categories)

- Quality of Life: 48
- Cultural Identity: 47
- Political: 43
- Capacity & Resources: 38
- Security & Defense: 35
- Legality, Constitutionality & Jurisprudence: 35
- Crime & Punishment: 34
- Public Opinion: 28
- Health & Safety: 24
- Policy Prescription & Evaluation: 22
- Fairness & Equality: 16
- Economic: 8
- Morality: 5

Never observed in this sample: external_regulation_and_reputation, other