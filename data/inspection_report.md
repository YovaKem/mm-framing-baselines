# Inspection report

Source: `sample_300.jsonl`  |  Rows: 300

## Columns

| column | non-null | unique | description |
|---|---|---|---|
| `uuid` | 300/300 | 300 | Unique ID for the article. |
| `title` | 300/300 | 300 | Article headline. |
| `date_publish` | 300/300 | 300 | Publication timestamp. |
| `source_domain` | 300/300 | 20 | Publisher's domain (28 unique outlets in the full dataset). |
| `url` | 300/300 | 300 | Original article URL — the only pointer back to the source text/image; not itself the content. |
| `political_leaning` | 300/300 | 5 | Publisher bias label: left / left_lean / center / right_lean / right. |
| `text-topic` | 300/300 | 252 | Main subject of the article (LLM-derived, short phrase). |
| `text-topic-exp` | 300/300 | 300 | LLM's free-text justification for the topic label. |
| `text-entity-name` | 299/300 | 270 | Key entity/person/org the article centers on (LLM-derived). |
| `text-entity-sentiment` | 295/300 | 7 | Sentiment expressed toward that entity in the text (LLM-derived). |
| `text-entity-sentiment-exp` | 300/300 | 300 | LLM's free-text justification for the entity sentiment. |
| `text-generic-frame` | 300/300 | 234 | MAIN LABEL — multi-label set of generic frames present in the article text, drawn from the 15-category Boydstun et al. taxonomy. Stored as a stringified Python list of short tags. |
| `text-generic-frame-exp` | 300/300 | 300 | LLM's free-text justification for the text frame labels. |
| `text-issue-frame` | 300/300 | 266 | Issue-specific (non-fixed-vocabulary) framing label for the text, e.g. 'Geopolitical Tension'. |
| `text-issue-frame-exp` | 300/300 | 300 | LLM's free-text justification for the issue frame label. |
| `img-generic-frame` | 300/300 | 54 | MAIN LABEL — multi-label set of generic frames present in the article's lead image, same 15-category taxonomy as text-generic-frame. Stringified Python list of short tags. |
| `img-frame-exp` | 300/300 | 300 | LLM's free-text justification for the image frame labels. |
| `img-entity-name` | 269/300 | 208 | Key entity/subject visible in the image (LLM-derived). |
| `img-entity-sentiment` | 295/300 | 5 | Sentiment conveyed by the image toward that entity (LLM-derived). |
| `img-entity-sentiment-exp` | 296/300 | 295 | LLM's free-text justification for the image entity sentiment. |
| `gpt-topic` | 300/300 | 36 | Separate GPT-generated topic classification (broader taxonomy than text-topic). |

## political_leaning distribution

political_leaning
left_lean     79
right         76
center        67
right_lean    53
left          25


## Top 15 gpt-topic values

gpt-topic
Politics         62
Crime            41
Business         32
Entertainment    24
Legal            22
Health           19
War              10
Culture          10
Economy           8
Immigration       8
Environment       6
Travel            6
no_topic          6
Labor             5
Lifestyle         5


## text-generic-frame: canonical tag frequency (of 15 taxonomy categories)

- Quality of Life: 121
- Policy Prescription & Evaluation: 117
- Legality, Constitutionality & Jurisprudence: 115
- Economic: 114
- Crime & Punishment: 111
- Political: 93
- Public Opinion: 73
- Fairness & Equality: 70
- Health & Safety: 67
- Cultural Identity: 64
- Security & Defense: 63
- External Regulation & Reputation: 38
- Capacity & Resources: 27
- Morality: 20

Never observed in this sample: other


## img-generic-frame: canonical tag frequency (of 15 taxonomy categories)

- Quality of Life: 46
- Cultural Identity: 46
- Political: 41
- Legality, Constitutionality & Jurisprudence: 38
- Security & Defense: 36
- Public Opinion: 33
- Health & Safety: 32
- Capacity & Resources: 31
- Policy Prescription & Evaluation: 28
- Crime & Punishment: 26
- Fairness & Equality: 17
- Economic: 7
- Morality: 6

Never observed in this sample: external_regulation_and_reputation, other