"""
Generate fresh text and image generic-frame labels from scratch, ignoring the
dataset's original labels, and store them side by side with the originals for
comparison (see scripts/report_overlap.py).

Per row, two independent sequential calls (text first, then image — the image
call is NOT shown the text call's own OUTPUT/labels, so any text/image
divergence is genuine signal rather than an artifact of anchoring):

  1. TEXT: given the article title + scraped text, return only frames that
     apply STRONGLY or MODERATELY — weak/tangential connections are dropped
     entirely, not just hidden.
  2. IMAGE: given the SAME full article title + text (as context/grounding)
     plus the image itself, judge which frames the IMAGE specifically
     visually conveys — not the article's topic in the abstract. Frames
     often overlap with the text's (the image usually illustrates the
     story) but this isn't required: an image can carry its own distinct
     framing the text never develops, or fail to visually convey a frame
     the text discusses. Many news images are purely illustrative (e.g. a
     plain storefront photo in a story about that store) and carry no
     framing at all — an EMPTY frame list ("None") is an explicitly valid,
     expected outcome, not a failure.

Requires OPENROUTER_API_KEY in .env. Uses anthropic/claude-haiku-4.5 by
default (common.OPENROUTER_MODEL).

Run after filter_news.py.

Usage:
    python scripts/relabel_frames.py
    python scripts/relabel_frames.py --workers 6
    python scripts/relabel_frames.py --image-only  # reuse existing text results, only redo image calls
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
    encode_image_b64,
    extract_json_object,
    read_jsonl,
    relabel_model_path,
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

You are given the full article text and its lead image. Read the article for context — who/what \
it's about, what's happening — but apply the frame labels specifically to what the IMAGE ITSELF \
visually conveys, not to the article's topic in the abstract. Image frames often overlap with the \
frames present in the text, since the image usually illustrates the story, but this is NOT a \
requirement: the image can carry its own distinct framing that the text never develops, or fail to \
visually convey a frame the text discusses. Use the article to understand what you're looking at, \
not as a source to copy frame labels from.
- strong: the image's composition/subject centrally conveys this frame.
- moderate: the image substantively supports this frame, but not centrally.
- weak (DO NOT INCLUDE): a stretch, or true only because of the article's topic rather than what's
  actually shown in the image.

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


def build_text_content(row):
    article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]
    return f"TITLE: {row.get('title', '')}\n\nTEXT:\n{article_text}"


def relabel_text_one(client, model, row):
    text_content = build_text_content(row)
    text_result = call_llm(client, model, TEXT_SYSTEM_PROMPT, text_content)
    return {
        "new_text_generic_frame": text_result["frames"],
        "new_text_generic_frame_strengths": text_result["frame_strengths"],
        "new_text_generic_frame_exp": text_result["explanation"],
        "new_text_generic_frame_error": text_result["error"],
    }


def relabel_image_one(client, model, row):
    text_content = build_text_content(row)
    image_path = DATA_DIR / row["image_local_path"]
    b64 = encode_image_b64(image_path)
    image_content = [
        {"type": "text", "text": text_content},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    image_result = call_llm(client, model, IMAGE_SYSTEM_PROMPT, image_content)
    return {
        "new_img_generic_frame": image_result["frames"],
        "new_img_generic_frame_strengths": image_result["frame_strengths"],
        "new_img_generic_frame_exp": image_result["explanation"],
        "new_img_generic_frame_error": image_result["error"],
    }


def relabel_one(client, model, row):
    result = {"uuid": row["uuid"]}
    result.update(relabel_text_one(client, model, row))
    result.update(relabel_image_one(client, model, row))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--image-only", action="store_true",
                         help="reuse existing text results from a prior run, only recompute image labels "
                              "(for when only the image prompt/content changed)")
    args = parser.parse_args()

    if not NEWS_SAMPLE_PATH.exists():
        raise SystemExit(f"{NEWS_SAMPLE_PATH} not found — run scripts/filter_news.py first")
    rows = read_jsonl(NEWS_SAMPLE_PATH)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out_path = relabel_model_path(args.model)

    if args.image_only:
        if not out_path.exists():
            raise SystemExit(f"--image-only requires an existing {out_path} to reuse text results from")
        existing_by_uuid = {r["uuid"]: r for r in read_jsonl(out_path)}

        results = []
        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(relabel_image_one, client, args.model, row): row for row in rows}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Re-imaging with {args.model}"):
                row = futures[future]
                image_res = future.result()
                merged = dict(existing_by_uuid[row["uuid"]])
                merged.update(image_res)
                results.append(merged)
                if image_res["new_img_generic_frame_error"]:
                    errors += 1
    else:
        results = []
        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(relabel_one, client, args.model, row): row["uuid"] for row in rows}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Relabeling with {args.model}"):
                res = future.result()
                results.append(res)
                if res["new_text_generic_frame_error"] or res["new_img_generic_frame_error"]:
                    errors += 1

    write_jsonl(out_path, results)
    print(f"\nRelabeled {len(results)} rows with {args.model} ({errors} had a call error after retries).")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
