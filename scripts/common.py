"""Shared paths, taxonomy, and parsing helpers for the mm-framing baseline scripts."""
import ast
import base64
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"

SAMPLE_PATH = DATA_DIR / "sample_300.jsonl"  # legacy: the old 300-article LLM-only sample (retired, not regenerated)
SCRAPE_ATTEMPTS_LOG_PATH = DATA_DIR / "scrape_attempts_log.jsonl"  # every row attempted, incl. rejected ones, for transparency
NEWS_FILTER_PATH = DATA_DIR / "news_filter.json"  # per-row is_news verdict + reason (legacy pipeline only)
NEWS_SAMPLE_PATH = DATA_DIR / "sample_news.jsonl"  # the sample every downstream script (relabel/baseline/finetune) actually reads
CONSOLIDATED_PATH = DATA_DIR / "sample_consolidated.jsonl"  # NEWS_SAMPLE_PATH + each model's labels + the consolidated target labels
CONSOLIDATION_REPORT_PATH = DATA_DIR / "consolidation_report.md"
FLAGS_PATH = DATA_DIR / "flags.json"
INSPECTION_REPORT_PATH = DATA_DIR / "inspection_report.md"

RANDOM_SEED = 42
SAMPLE_SIZE = 300  # legacy pipeline only (build_sample.py)
MIN_ARTICLE_WORDS = 100  # matches the paper's own filtering threshold

HF_DATASET = "copenlu/mm-framing"
HF_SPLIT = "valid_framing_subset"  # the paper's framing-analysis-ready subset (154k rows) — legacy pipeline only
HF_FULL_DATA_FILE = "annotated_data.csv"  # the full, unfiltered split (~479k rows) — has the widest uuid coverage

# The paper's own human double-annotation pass over images (docs/DATASET.md), used as
# the image-frame ground truth for the current sample (see scripts/build_human_sample.py).
HUMAN_LABELS_CSV_PATH = DATA_DIR / "human_image_frame_labels.csv"

# Of the current sample, a fixed split: TEST rows are held out of every finetune's
# training data and are the only rows any baseline (zero-shot or finetuned) is ever
# scored against; TRAIN_LORA rows are training data for the two LoRA finetunes only.
# (Originally 350/228, matching the full 578-uuid sample; scaled down to 300/100 once
# scraping showed only ~400/578 articles are actually recoverable — see
# docs/PIPELINE.md's sampling caveat. CBS News alone is 46% of the 578 and is largely
# unrecoverable, live or via the Wayback Machine.)
SPLIT_TEST_SIZE = 300
SPLIT_TRAIN_LORA_SIZE = 100

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

# The 15 generic frames from Boydstun et al. (2014) / Media Frames Corpus, as used
# for BOTH text-generic-frame and img-generic-frame in this dataset (see arXiv:2503.20960).
# The 15th category is "None" (a real, promptable/selectable taxonomy member), matching
# the paper's own text- and image-framing prompts exactly — NOT "Other", which this repo
# used before this taxonomy was cross-checked against the paper's actual prompt listings.
# Display names here are the single source of truth for capitalization: both
# TEXT_FRAME_DESCRIPTIONS/IMAGE_FRAME_DESCRIPTIONS below and every prompt built from them
# use these exact names, so a model's output can only ever need to match one spelling.
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
    "none": "None — no frame applies",
}

# Per-frame description text used to build the two LLM system prompts, adapted verbatim
# from the paper's own text- and image-framing prompts (arXiv:2503.20960 PDF, pp.18-19)
# — these are genuinely different per modality (image descriptions are visually grounded,
# text descriptions are terse). Keyed the same as CANONICAL_FRAMES; display name for each
# frame in the actual prompt text always comes from CANONICAL_FRAMES, not from here, so
# capitalization stays consistent between the two prompts and with the parser.
TEXT_FRAME_DESCRIPTIONS = {
    "economic": "costs, benefits, or other financial implications",
    "capacity_and_resources": "availability of physical, human, or financial resources, and capacity of current systems",
    "morality": "religious or ethical implications",
    "fairness_and_equality": "balance or distribution of rights, responsibilities, and resources",
    "legality_constitutionality_jurisprudence": "rights, freedoms, and authority of individuals, corporations, and government",
    "policy_prescription_and_evaluation": "discussion of specific policies aimed at addressing problems",
    "crime_and_punishment": "effectiveness and implications of laws and their enforcement",
    "security_and_defense": "threats to welfare of the individual, community, or nation",
    "health_and_safety": "health care, sanitation, public safety",
    "quality_of_life": "threats and opportunities for the individual's wealth, happiness, and well-being",
    "cultural_identity": "traditions, customs, or values of a social group in relation to a policy issue",
    "public_opinion": "attitudes and opinions of the general public, including polling and demographics",
    "political": "considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters",
    "external_regulation_and_reputation": "international reputation or foreign policy of the U.S",
    "none": "none of the above or any frame not covered by the above categories",
}

IMAGE_FRAME_DESCRIPTIONS = {
    "economic": (
        "costs, benefits, or other finance related. The image can include things including but not "
        "limited to money, funding, taxes, bank, meetings with a logo of a financial institution. If "
        "you are using a logo of a financial institution to classify it as economic, make sure it is "
        "clearly visible. If it is not clearly visible, it should be classified as 'None'. Professional "
        "attire in itself does not mean economic frame."
    ),
    "capacity_and_resources": (
        "availability of physical, human, or financial resources, and capacity of current systems. In "
        "the image, we can see things including but not limited to a geographical area, farmland, "
        "agriculture land, labour, people working in an institution, or images that convey scarcity or "
        "surplus in some way."
    ),
    "morality": (
        "religious or ethical implications. In the image, we can see things including but not limited "
        "to god, death, priests, church, protests related to moral issues."
    ),
    "fairness_and_equality": (
        "balance or distribution of rights, responsibilities, and resources. In the image, we can see "
        "things including but not limited to the fight for civil or political rights, LGBTQ, or calls "
        "to stop discrimination."
    ),
    "legality_constitutionality_jurisprudence": (
        "legal rights, freedoms, and authority of individuals, corporations, and government. In the "
        "image, we can see things including but not limited to prisons, laws, judges in robes, "
        "courtrooms, legal documents, and prison facilities. This does not include sports contexts, "
        "such as referees or players enforcing or breaking game rules."
    ),
    "policy_prescription_and_evaluation": (
        "discussion of specific policies aimed at addressing problems. In the image, we can see things "
        "including but not limited to discussions on rule-making bodies, people in formal settings such "
        "as boardrooms or legislative halls actively debating and reviewing policy drafts or proposals. "
        "You might see official charts, graphs, or official documents. People in formal attire with no "
        "other information should not be classified as policy prescription and evaluation."
    ),
    "crime_and_punishment": (
        "effectiveness and implications of laws and their enforcement. In the image, we can see things "
        "including but not limited to criminal activities, violence, police officers making arrests, "
        "crime scenes with investigators, courtrooms during criminal trials, prisons with detainees. "
        "This frame specifically excludes contexts involving sports, such as referees, players, or rule "
        "enforcement within games, which are not related to societal law violations or legal punishment."
    ),
    "security_and_defense": (
        "threats to the individual, community, or nation. In the image, we can see things including but "
        "not limited to military uniforms, defense personnel, border patrol, war, soldiers, military "
        "equipment like tanks or fighter jets, border walls, or surveillance systems monitoring wide "
        "areas."
    ),
    "health_and_safety": (
        "health care, sanitation, public safety. Images with objects like coffee, drinks, food items or "
        "activities like sports which give a clear and literal message that it affects health and "
        "safety positively or negatively should be classified as health and safety, otherwise it should "
        "be classified as 'None'. E.g. a person drinking coffee does not mean health and safety, but a "
        "person drinking medicine or having a cigarette does. A bus does not mean health and safety, but "
        "a bus with a warning sign does. In the image, we can see things including but not limited to "
        "doctors, nurses, injury, disease, or events with environmental impact that may impact health "
        "and safety."
    ),
    "quality_of_life": (
        "threats and opportunities for the individual's wealth, happiness, and well-being. In the "
        "image, we can see things that improve happiness or demonstrate quality of life in some form. "
        "It also includes things that demonstrate deterioration of quality of life by showing hardships "
        "of people, homelessness, etc. This may also include happy children, food items that "
        "demonstrate good quality of life, or people enjoying a nice meal."
    ),
    "cultural_identity": (
        "traditions, customs, or values of a social group in relation to a policy issue. In the image, "
        "we can see things including but not limited to concerts, cultural dance, sports, art, "
        "celebrities, artists and prominent people related to these topics. Examples: celebrities, "
        "traditional dress, sports with clear country-specific detail e.g. jerseys/flags, cultural "
        "events, cultural art, etc. Otherwise, it should be classified as 'None'."
    ),
    "public_opinion": (
        "attitudes and opinions of the general public, including polling and demographics. Includes "
        "generic protests, people (non-celebrities) engaging with large crowds, riots, and strikes, and "
        "including but not limited to sharing petitions and encouraging people to take political "
        "action. It will also include news broadcasts, talk shows, and interviews with people that are "
        "related to public opinion at large."
    ),
    "political": (
        "considerations related to politics and politicians, including lobbying, elections, and "
        "attempts to sway voters. In the image, we can see things related to politicians, elections, "
        "voting, political campaigns. Just formal clothing does not mean political frame. If the image "
        "does not have a recognizable political person, it should not be classified as political. "
        "Formal attire with no political information should be classified as 'None'."
    ),
    "external_regulation_and_reputation": (
        "international reputation or foreign policy. In the image, we can see things including but not "
        "limited to international organizations, global discussions/meetings, foreign policy, flags "
        "from multiple countries, or delegates at a cross-country forum discussing reputation and "
        "regulation. If you use a logo of a global organization to classify it as external regulation "
        "and reputation, make sure it is clearly visible in the image. If it is not clearly visible, it "
        "should be classified as 'None'."
    ),
    "none": (
        "no frame could be identified because of lack of information in the image. This should be "
        "selected when no other frame is applicable. Example: a handshake with no other information, a "
        "logo of a company with no other information, a landscape with no other information, a person "
        "in a photo album with no other information, a person speaking with no other information about "
        "the content of the speech or person's identity, a formal event with no other information, a "
        "person in formal attire with no other information, a news logo with no news, a sports event "
        "with no additional information, simple objects like vehicle/car/pen/paper/sign-boards/objects, "
        "etc. with no other information, etc."
    ),
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
    "none": "none",
    # Full space-separated frame names, as literally used by
    # data/human_image_frame_labels.csv's merged_human_frames column — parse_list_field
    # lowercases/strips each tag but never collapses spaces to underscores, so these need
    # their own entries (the single-word aliases above don't cover them).
    "quality of life": "quality_of_life",
    "health and safety": "health_and_safety",
    "capacity and resources": "capacity_and_resources",
    "cultural identity": "cultural_identity",
    "fairness and equality": "fairness_and_equality",
    "policy prescription and evaluation": "policy_prescription_and_evaluation",
    "crime and punishment": "crime_and_punishment",
    "security and defense": "security_and_defense",
    "public opinion": "public_opinion",
    "external regulation and reputation": "external_regulation_and_reputation",
}

# Reverse lookup used when parsing model output that names frames by their display
# label (e.g. "Capacity & Resources") rather than the internal snake_case key.
# Lowercased keys: the paper's own text- and image-framing prompts don't capitalize
# every frame name identically to each other (e.g. "Public Opinion" vs "Public opinion"),
# so lookups are done case-insensitively as a defensive backstop — the prompts themselves
# are still built from CANONICAL_FRAMES's one consistent set of display names.
CANONICAL_LABEL_TO_KEY = {v.split(" — ")[0].lower(): k for k, v in CANONICAL_FRAMES.items()}

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


def strip_none_key(keys):
    """Drop the literal 'none' taxonomy key from a frame-key list. "None" is a real,
    promptable/parseable taxonomy member (models can select it, and the human CSV's
    'none' tag parses cleanly against FRAME_TAG_ALIASES), but every downstream
    consumer (consolidation, baselines, scoring) represents "no frame applies" as an
    empty list, not as a member tag — so this is applied right after parsing, at every
    boundary (CSV rows, LLM responses), to keep that convention uniform."""
    return [k for k in keys if k != "none"]


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
        key = CANONICAL_LABEL_TO_KEY.get(name.strip().lower())
        if key is None:
            unrecognized.append(name)
        else:
            keys.append(key)
    return keys, unrecognized
