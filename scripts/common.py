"""Shared paths, taxonomy, and parsing helpers for the mm-framing baseline scripts."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"

SAMPLE_PATH = DATA_DIR / "sample_300.jsonl"  # 300 rows, each with BOTH article text and image successfully scraped
SCRAPE_ATTEMPTS_LOG_PATH = DATA_DIR / "scrape_attempts_log.jsonl"  # every row attempted, incl. rejected ones, for transparency
FLAGS_PATH = DATA_DIR / "flags.json"
FRAME_JUDGMENTS_PATH = DATA_DIR / "frame_judgments.json"
VALIDATION_RESULTS_PATH = DATA_DIR / "validation_results.csv"
INSPECTION_REPORT_PATH = DATA_DIR / "inspection_report.md"

RANDOM_SEED = 42
SAMPLE_SIZE = 300
MIN_ARTICLE_WORDS = 100  # matches the paper's own filtering threshold

HF_DATASET = "copenlu/mm-framing"
HF_SPLIT = "valid_framing_subset"  # the paper's framing-analysis-ready subset (154k rows)

OPENROUTER_JUDGE_MODEL = "anthropic/claude-haiku-4.5"

# The 15 generic frames from Boydstun et al. (2014) / Media Frames Corpus, as used
# for BOTH text-generic-frame and img-generic-frame in this dataset (see arXiv:2503.20960).
CANONICAL_FRAMES = {
    "economic": "Economic — costs, benefits, or monetary/financial implications",
    "capacity_and_resources": "Capacity & Resources — lack of or availability of physical, geographical, spatial, human, and financial resources",
    "morality": "Morality — perspective compelled by religious doctrine or interpretation, duty, honor, righteousness",
    "fairness_and_equality": "Fairness & Equality — equality or inequality with which laws, punishment, rewards, and resources are applied",
    "legality_constitutionality_jurisprudence": "Legality, Constitutionality & Jurisprudence — constraints imposed on or freedoms granted via Constitution and judicial interpretation",
    "policy_prescription_and_evaluation": "Policy Prescription & Evaluation — particular policies proposed for addressing an identified problem",
    "crime_and_punishment": "Crime & Punishment — enforcement and interpretation of laws, breaking laws, sentencing and punishment",
    "security_and_defense": "Security & Defense — security, threats to security, and protection of one's person, family, nation",
    "health_and_safety": "Health & Safety — healthcare access, illness, disease, sanitation, prevention of gun violence",
    "quality_of_life": "Quality of Life — effects of policy on individuals' wealth, mobility, access to resources, happiness",
    "cultural_identity": "Cultural Identity — social norms, trends, values and customs constituting culture(s)",
    "public_opinion": "Public Opinion — general social attitudes, polling and demographic information",
    "political": "Political — partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making",
    "external_regulation_and_reputation": "External Regulation & Reputation — a country's external relations with another nation; trade agreements",
    "other": "Other — frames that do not fit into the above categories",
}

# Best-effort mapping from the short-form tags actually observed in the raw CSV
# (e.g. "security", "legality", "regulation", "policy") to the canonical keys above.
# Any tag NOT covered here gets flagged by flag_issues.py as "unknown_frame_tag" so
# it surfaces for manual review rather than silently passing through.
FRAME_TAG_ALIASES = {
    "economic": "economic",
    "economy": "economic",
    "capacity": "capacity_and_resources",
    "resources": "capacity_and_resources",
    "capacity_and_resources": "capacity_and_resources",
    "morality": "morality",
    "fairness": "fairness_and_equality",
    "equality": "fairness_and_equality",
    "fairness_and_equality": "fairness_and_equality",
    "legality": "legality_constitutionality_jurisprudence",
    "constitutionality": "legality_constitutionality_jurisprudence",
    "jurisprudence": "legality_constitutionality_jurisprudence",
    "policy": "policy_prescription_and_evaluation",
    "policy_prescription": "policy_prescription_and_evaluation",
    "crime": "crime_and_punishment",
    "punishment": "crime_and_punishment",
    "crime_and_punishment": "crime_and_punishment",
    "security": "security_and_defense",
    "defense": "security_and_defense",
    "security_and_defense": "security_and_defense",
    "health": "health_and_safety",
    "safety": "health_and_safety",
    "health_and_safety": "health_and_safety",
    "quality": "quality_of_life",
    "quality_life": "quality_of_life",
    "quality_of_life": "quality_of_life",
    "cultural": "cultural_identity",
    "culture": "cultural_identity",
    "identity": "cultural_identity",
    "cultural_identity": "cultural_identity",
    "public_opinion": "public_opinion",
    "public_op": "public_opinion",
    "opinion": "public_opinion",
    "political": "political",
    "politics": "political",
    "regulation": "external_regulation_and_reputation",
    "reputation": "external_regulation_and_reputation",
    "external_regulation": "external_regulation_and_reputation",
    "cap&res": "capacity_and_resources",
    "other": "other",
}

ENTITY_SENTIMENT_VALUES = {"positive", "negative", "neutral"}
POLITICAL_LEANING_VALUES = {"left", "left_lean", "center", "right_lean", "right"}


def parse_list_field(value):
    """Parse a stringified Python list column (e.g. "['security', 'policy']") into a
    real list of lowercased, stripped tags. Returns (parsed_list, error) where error is
    None on success or a short description of what went wrong."""
    if value is None:
        return [], "missing"
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()], None
    text = str(value).strip()
    if not text:
        return [], "empty_string"
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [], f"malformed_list_literal: {text[:80]!r}"
    if not isinstance(parsed, list):
        return [], f"not_a_list: {type(parsed).__name__}"
    return [str(v).strip().lower() for v in parsed if str(v).strip()], None


def normalize_frame_tags(tags):
    """Map raw short-form tags to canonical taxonomy keys. Returns (canonical, unknown)."""
    canonical, unknown = [], []
    for tag in tags:
        key = FRAME_TAG_ALIASES.get(tag)
        if key is None:
            unknown.append(tag)
        else:
            canonical.append(key)
    return canonical, unknown


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
