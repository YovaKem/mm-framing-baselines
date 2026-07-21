"""Shared paths, taxonomy, and parsing helpers for the mm-framing baseline scripts."""
import ast
import base64
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"

SAMPLE_PATH = DATA_DIR / "sample_300.jsonl"  # 300 rows, each with BOTH article text and image successfully scraped
SCRAPE_ATTEMPTS_LOG_PATH = DATA_DIR / "scrape_attempts_log.jsonl"  # every row attempted, incl. rejected ones, for transparency
NEWS_FILTER_PATH = DATA_DIR / "news_filter.json"  # per-row is_news verdict + reason
NEWS_SAMPLE_PATH = DATA_DIR / "sample_news.jsonl"  # SAMPLE_PATH minus rows filtered out as non-news
RELABELED_PATH = DATA_DIR / "sample_relabeled.jsonl"  # NEWS_SAMPLE_PATH + single-model (claude-haiku-4.5) fresh labels — superseded by the 3-model ensemble below, kept for the overlap-vs-original report
OVERLAP_REPORT_PATH = DATA_DIR / "overlap_report.md"
CONSOLIDATED_PATH = DATA_DIR / "sample_consolidated.jsonl"  # NEWS_SAMPLE_PATH + each model's labels + the 2-of-3 majority-vote consolidated labels
CONSOLIDATION_REPORT_PATH = DATA_DIR / "consolidation_report.md"
FLAGS_PATH = DATA_DIR / "flags.json"
VALIDATION_RESULTS_PATH = DATA_DIR / "validation_results.csv"
INSPECTION_REPORT_PATH = DATA_DIR / "inspection_report.md"

RANDOM_SEED = 42
SAMPLE_SIZE = 300
MIN_ARTICLE_WORDS = 100  # matches the paper's own filtering threshold

HF_DATASET = "copenlu/mm-framing"
HF_SPLIT = "valid_framing_subset"  # the paper's framing-analysis-ready subset (154k rows)

OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"

# 3-model ensemble for consolidated relabeling: one from each of three distinct
# training/RLHF pipelines, all at a comparable cheap/fast cost tier so the comparison
# isn't skewed by one model being much bigger or pricier than the others.
ENSEMBLE_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "google/gemini-3.5-flash",
]
AGREEMENT_THRESHOLD = 2  # keep a frame if >= this many of len(ENSEMBLE_MODELS) models assign it

FRAME_STRENGTHS = ["strong", "moderate"]  # "weak" frames are deliberately discarded, not just hidden

# The 15 generic frames from Boydstun et al. (2014) / Media Frames Corpus, as used
# for BOTH text-generic-frame and img-generic-frame in this dataset (see arXiv:2503.20960).
# Definitions are the codebook's own wording, verbatim (not a paraphrase) — passed to the
# LLM annotator in full so it judges against the actual definition, not just the surface label.
CANONICAL_FRAMES = {
    "economic": "Economic — costs, benefits, or other financial implications",
    "capacity_and_resources": "Capacity & Resources — availability of physical, human or financial resources, and capacity of current systems",
    "morality": "Morality — religious or ethical implications, considerations, issues, etc.",
    "fairness_and_equality": "Fairness & Equality — balance or distribution of rights, responsibilities, and resources",
    "legality_constitutionality_jurisprudence": "Legality, Constitutionality & Jurisprudence — rights, freedoms, and authority of individuals, corporations, and government",
    "policy_prescription_and_evaluation": "Policy Prescription & Evaluation — discussion of specific policies aimed at addressing problems, needs, issues, etc.",
    "crime_and_punishment": "Crime & Punishment — effectiveness and implications of laws and their enforcement",
    "security_and_defense": "Security & Defense — threats to welfare of the individual, community, or nation",
    "health_and_safety": "Health & Safety — health care, sanitation, public safety",
    "quality_of_life": "Quality of Life — threats and opportunities for the individual's wealth, happiness, and well-being",
    "cultural_identity": "Cultural Identity — traditions, customs, or values of a social group in relation to a policy issue",
    "public_opinion": "Public Opinion — attitudes and opinions of the general public, including polling and demographics",
    "political": "Political — considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters",
    "external_regulation_and_reputation": "External Regulation & Reputation — international reputation or foreign policy of the U.S.",
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

# Reverse lookup used when parsing model output that names frames by their display
# label (e.g. "Capacity & Resources") rather than the internal snake_case key.
CANONICAL_LABEL_TO_KEY = {v.split(" — ")[0]: k for k, v in CANONICAL_FRAMES.items()}

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


def extract_json_object(text):
    """Parse the first complete JSON object out of an LLM response, ignoring any
    markdown fences or trailing text/stray characters after it (models occasionally
    emit an extra closing brace or follow-up commentary, which breaks a naive
    greedy-regex-then-json.loads approach on the "Extra data" trailing bytes)."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.JSONDecoder().raw_decode(text, start)[0]


def encode_image_b64(path, max_dim=768):
    """Downscale + JPEG-encode an image file to a base64 string, for vision API calls."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def model_slug(model):
    return model.replace("/", "_").replace(".", "_")


def relabel_model_path(model):
    """Per-model relabeling output: data/relabel_<slug>.jsonl, just the uuid + label
    fields (not the full row — the base rows are already in NEWS_SAMPLE_PATH)."""
    return DATA_DIR / f"relabel_{model_slug(model)}.jsonl"


def frames_to_keys(frame_names):
    """Map model-output frame display names (e.g. 'Capacity & Resources') to canonical
    keys. Returns (keys, unrecognized) — unrecognized names are kept as-is for visibility."""
    keys, unrecognized = [], []
    for name in frame_names:
        key = CANONICAL_LABEL_TO_KEY.get(name.strip())
        if key is None:
            unrecognized.append(name)
        else:
            keys.append(key)
    return keys, unrecognized
