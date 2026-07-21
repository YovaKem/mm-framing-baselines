"""
Generate fresh text and image generic-frame labels from scratch, ignoring the
dataset's original labels, and store them side by side with the originals for
comparison (see scripts/report_overlap.py).

Per row, two independent sequential calls (text first, then image — the image
call is NOT shown the text call's output, so any text/image divergence is
genuine signal rather than an artifact of anchoring):

  1. TEXT: given the article title + scraped text, return only frames that
     apply STRONGLY or MODERATELY — weak/tangential connections are dropped
     entirely, not just hidden.
  2. IMAGE: given the image + article title (for minimal subject grounding),
     same strong/moderate-only rule. Many news images are purely illustrative
     (e.g. a plain storefront photo in a story about that store) and carry no
     framing at all — an EMPTY frame list ("None") is an explicitly valid,
     expected outcome, not a failure.

Requires OPENROUTER_API_KEY in .env. Uses anthropic/claude-haiku-4.5 by
default (common.OPENROUTER_MODEL).

Run after filter_news.py.

Usage:
    python scripts/relabel_frames.py
    python scripts/relabel_frames.py --workers 6
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from common import (
    CANONICAL_FRAMES,
    CANONICAL_LABEL_TO_KEY,
    DATA_DIR,
    NEWS_SAMPLE_PATH,
    OPENROUTER_MODEL,
    RELABELED_PATH,
    encode_image_b64,
    extract_json_object,
    read_jsonl,
    write_jsonl,
)

load_dotenv()

MAX_ARTICLE_CHARS = 6000
FRAME_LIST_BLOCK = "\n".join(f"- {v}" for v in CANONICAL_FRAMES.values())

TEXT_SYSTEM_PROMPT = f"""You are a media framing analyst labeling news articles with a \
fixed taxonomy (Boydstun et al. / Media Frames Corpus):

{FRAME_LIST_BLOCK}

Given the article's title and text, decide which frames are STRONGLY or MODERATELY \
present as a way the article frames its subject:
- strong: central to how the article is written — a reader would name this as one of the
  main angles.
- moderate: clearly and substantively present, but not the main angle.
- weak (DO NOT INCLUDE AT ALL): only mentioned in passing, or arguable but not
  substantively developed. Do not pad the list with weak matches — it is normal and often
  correct to return just one or two frames, even a single one.

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{"frames": [{{"frame": "<exact name from the list above>", "strength": "strong"|"moderate"}}, ...], \
"explanation": "<1-3 sentences justifying the selected set as a whole>"}}"""

IMAGE_SYSTEM_PROMPT = f"""You are a media framing analyst labeling the LEAD IMAGE of news \
articles with a fixed taxonomy (Boydstun et al. / Media Frames Corpus), applied visually:

{FRAME_LIST_BLOCK}

You are given the image and the article's title (for minimal subject context only). Decide \
which frames the IMAGE ITSELF visually conveys STRONGLY or MODERATELY — not what the
article's topic is about in the abstract, only what's actually depicted:
- strong: the image's composition/subject centrally conveys this frame.
- moderate: the image substantively supports this frame, but not centrally.
- weak (DO NOT INCLUDE): a stretch, or true only because of the topic rather than what's
  actually shown.

Many news images are purely illustrative/neutral (a plain storefront photo, a generic stock
photo, a headshot, a building exterior) and convey NO editorial frame at all — in that case
return an EMPTY frames list. This is a common, entirely valid, EXPECTED outcome, not a
failure to find something — do not force a weak match just to return something.

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{"frames": [{{"frame": "<exact name from the list above>", "strength": "strong"|"moderate"}}, ...], \
"explanation": "<1-3 sentences; if empty, briefly say why the image is neutral/illustrative>"}}"""


def call_llm(client, model, system_prompt, content, retries=2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
                temperature=0,
            )
            parsed = extract_json_object(resp.choices[0].message.content)
            frames_raw = parsed.get("frames", [])
            keys, frame_strengths, unrecognized = [], {}, []
            for f in frames_raw:
                name = str(f.get("frame", "")).strip()
                key = CANONICAL_LABEL_TO_KEY.get(name)
                if key is None:
                    unrecognized.append(name)
                else:
                    keys.append(key)
                    frame_strengths[key] = f.get("strength", "")
            return {
                "frames": keys,
                "frame_strengths": frame_strengths,
                "unrecognized_frame_names": unrecognized,
                "explanation": parsed.get("explanation", ""),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))
    return {"frames": [], "frame_strengths": {}, "unrecognized_frame_names": [], "explanation": None,
            "error": f"relabel_failed_after_retries: {last_error}"}


def relabel_one(client, model, row):
    article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]
    text_content = f"TITLE: {row.get('title', '')}\n\nTEXT:\n{article_text}"
    text_result = call_llm(client, model, TEXT_SYSTEM_PROMPT, text_content)

    image_path = DATA_DIR / row["image_local_path"]
    b64 = encode_image_b64(image_path)
    image_content = [
        {"type": "text", "text": f"ARTICLE TITLE (for context only): {row.get('title', '')}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    image_result = call_llm(client, model, IMAGE_SYSTEM_PROMPT, image_content)

    return {
        "uuid": row["uuid"],
        "new_text_generic_frame": text_result["frames"],
        "new_text_generic_frame_strengths": text_result["frame_strengths"],
        "new_text_generic_frame_exp": text_result["explanation"],
        "new_text_generic_frame_error": text_result["error"],
        "new_img_generic_frame": image_result["frames"],
        "new_img_generic_frame_strengths": image_result["frame_strengths"],
        "new_img_generic_frame_exp": image_result["explanation"],
        "new_img_generic_frame_error": image_result["error"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if not NEWS_SAMPLE_PATH.exists():
        raise SystemExit(f"{NEWS_SAMPLE_PATH} not found — run scripts/filter_news.py first")
    rows = read_jsonl(NEWS_SAMPLE_PATH)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    results_by_uuid = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(relabel_one, client, args.model, row): row["uuid"] for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Relabeling with {args.model}"):
            res = future.result()
            results_by_uuid[res["uuid"]] = res
            if res["new_text_generic_frame_error"] or res["new_img_generic_frame_error"]:
                errors += 1

    merged = []
    for row in rows:
        row = dict(row)
        row.update(results_by_uuid[row["uuid"]])
        merged.append(row)

    write_jsonl(RELABELED_PATH, merged)
    print(f"\nRelabeled {len(merged)} rows ({errors} had a call error after retries).")
    print(f"Wrote {RELABELED_PATH}")


if __name__ == "__main__":
    main()
