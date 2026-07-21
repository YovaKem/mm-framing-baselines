"""
Semantic accuracy check: for each of the 300 sampled rows, show a vision LLM
the actual scraped article text + image, the frame label(s) an earlier LLM
assigned (text-generic-frame / img-generic-frame), and that earlier LLM's own
justification — then ask it to independently judge whether the label is
actually well-supported by the content (not just whether the justification
*sounds* plausible).

Requires OPENROUTER_API_KEY in a local .env file (git-ignored, never commit
this). Uses anthropic/claude-haiku-4.5 via OpenRouter by default — vision-
capable and cheap enough for a few hundred rows; override with
--model or common.OPENROUTER_JUDGE_MODEL.

Run after build_sample.py.

Usage:
    python scripts/judge_frames.py
    python scripts/judge_frames.py --workers 5 --model anthropic/claude-haiku-4.5
"""
import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from common import (
    CANONICAL_FRAMES,
    DATA_DIR,
    FRAME_JUDGMENTS_PATH,
    OPENROUTER_JUDGE_MODEL,
    SAMPLE_PATH,
    normalize_frame_tags,
    read_jsonl,
)

load_dotenv()

MAX_ARTICLE_CHARS = 4000
MAX_IMAGE_DIM = 768
REQUIRED_KEYS = {"text_frame_verdict", "text_frame_reasoning", "img_frame_verdict", "img_frame_reasoning"}

TAXONOMY_BLOCK = "\n".join(f"- {v}" for v in CANONICAL_FRAMES.values())

SYSTEM_PROMPT = f"""You are an expert media-framing analyst doing quality control on an \
automatically-labeled news framing dataset. Frames come from this fixed taxonomy:

{TAXONOMY_BLOCK}

You will see a news article's text and lead image, the generic frame label(s) an \
earlier AI system assigned to the TEXT and separately to the IMAGE, and that \
system's own justification for each. Judge independently whether each label set is \
actually well-supported by the content itself — a justification can sound fluent \
without actually supporting the label it's attached to, or can cite things not \
really present in the text/image. Flag a label as "questionable" if it's wrong, \
missing an obviously present frame, or the justification doesn't hold up; use \
"accurate" otherwise.

Respond with ONLY a JSON object (no markdown fences, no extra text) with exactly \
these keys: text_frame_verdict ("accurate" or "questionable"), text_frame_reasoning \
(one or two sentences), img_frame_verdict ("accurate" or "questionable"), \
img_frame_reasoning (one or two sentences)."""


def canonical_labels(tags):
    canonical, unknown = normalize_frame_tags(tags)
    labels = [CANONICAL_FRAMES[c].split(" — ")[0] for c in canonical]
    return labels + [f"{t} (unrecognized tag)" for t in unknown]


def encode_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(row):
    text_labels = canonical_labels(row.get("_text_generic_frame_parsed") or [])
    img_labels = canonical_labels(row.get("_img_generic_frame_parsed") or [])
    article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]

    user_text = (
        f"ARTICLE TITLE: {row.get('title', '')}\n\n"
        f"ARTICLE TEXT:\n{article_text}\n\n"
        f"---\n"
        f"ASSIGNED TEXT FRAME(S): {', '.join(text_labels) or '(none)'}\n"
        f"JUSTIFICATION GIVEN: {row.get('text-generic-frame-exp', '')}\n\n"
        f"ASSIGNED IMAGE FRAME(S) for the attached image: {', '.join(img_labels) or '(none)'}\n"
        f"JUSTIFICATION GIVEN: {row.get('img-frame-exp', '')}"
    )

    content = [{"type": "text", "text": user_text}]
    image_path = DATA_DIR / row["image_local_path"]
    b64 = encode_image(image_path)
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def extract_json(text):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def judge_one(client, model, row, retries=2):
    uuid = row["uuid"]
    last_error = None
    for attempt in range(retries + 1):
        try:
            messages = build_messages(row)
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            parsed = extract_json(resp.choices[0].message.content)
            if not REQUIRED_KEYS.issubset(parsed.keys()):
                raise ValueError(f"Missing keys in response: {parsed.keys()}")
            parsed["uuid"] = uuid
            parsed["error"] = None
            return parsed
        except Exception as e:  # noqa: BLE001 - want to retry/record any failure uniformly
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))
    return {
        "uuid": uuid,
        "text_frame_verdict": None,
        "text_frame_reasoning": None,
        "img_frame_verdict": None,
        "img_frame_reasoning": None,
        "error": f"judge_failed_after_retries: {last_error}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OPENROUTER_JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    if not SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found — run scripts/build_sample.py first")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    rows = read_jsonl(SAMPLE_PATH)
    judgments = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, client, args.model, row): row["uuid"] for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Judging with {args.model}"):
            result = future.result()
            judgments[result["uuid"]] = result
            if result.get("error"):
                errors += 1

    FRAME_JUDGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRAME_JUDGMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False)

    questionable_text = sum(1 for j in judgments.values() if j.get("text_frame_verdict") == "questionable")
    questionable_img = sum(1 for j in judgments.values() if j.get("img_frame_verdict") == "questionable")
    print(f"\nJudged {len(judgments)} rows ({errors} failed after retries).")
    print(f"  text-generic-frame flagged questionable: {questionable_text}")
    print(f"  img-generic-frame flagged questionable: {questionable_img}")
    print(f"Wrote {FRAME_JUDGMENTS_PATH}")


if __name__ == "__main__":
    main()
