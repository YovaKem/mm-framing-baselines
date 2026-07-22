"""
Generate fresh text generic-frame labels from scratch, ignoring the dataset's
original labels. Run once per model in common.ENSEMBLE_MODELS;
scripts/consolidate_annotations.py combines the results into a single target
label per row (2-of-3 majority vote) — the silver-standard TEXT ground truth
(no human text-frame annotation exists, unlike image frames, which come from
the paper's own human double-annotation pass — see scripts/build_human_sample.py
and scripts/consolidate_annotations.py).

TEXT_SYSTEM_PROMPT/IMAGE_SYSTEM_PROMPT follow the paper's own text- and
image-framing prompts verbatim (arXiv:2503.20960 PDF, pp.18-19; per-frame
description text lives in common.TEXT_FRAME_DESCRIPTIONS/
IMAGE_FRAME_DESCRIPTIONS), dropping the paper's topic/entity/issue-specific-
frame sub-tasks (out of scope here — generic frame only) and its
strong/moderate/weak strength grading (the paper's prompts don't have it).
Two deliberate deviations from the paper: (1) the image prompt is given the
full article title+text as context, even though the paper's own image prompt
is image-only; (2) both prompts are also used, unchanged, by the Qwen
baseline/finetune scripts (baseline_qwen_text.py, baseline_qwen_vlm.py,
finetune_qwen_text.py, finetune_qwen_vlm.py) so every baseline is scored
against ground truth generated under the exact prompt it's run with.

Requires OPENROUTER_API_KEY in .env. Uses anthropic/claude-haiku-4.5 by
default (common.OPENROUTER_MODEL).

Run after scripts/build_human_sample.py (no news-filtering step for this
sample — see that script's docstring).

Usage:
    python scripts/relabel_frames.py
    python scripts/relabel_frames.py --model openai/gpt-5.4-mini --workers 6
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from common import (
    CANONICAL_FRAMES,
    CANONICAL_LABEL_TO_KEY,
    IMAGE_FRAME_DESCRIPTIONS,
    NEWS_SAMPLE_PATH,
    OPENROUTER_MODEL,
    TEXT_FRAME_DESCRIPTIONS,
    extract_json_object,
    read_jsonl,
    relabel_model_path,
    strip_none_key,
    write_jsonl,
)

load_dotenv()

MAX_ARTICLE_CHARS = 6000


def frame_list_block(descriptions):
    display_name = lambda k: CANONICAL_FRAMES[k].split(" — ")[0]  # noqa: E731
    return "\n".join(f"{display_name(k)} - {descriptions[k]}" for k in CANONICAL_FRAMES)


TEXT_FRAME_LIST_BLOCK = frame_list_block(TEXT_FRAME_DESCRIPTIONS)
IMAGE_FRAME_LIST_BLOCK = frame_list_block(IMAGE_FRAME_DESCRIPTIONS)

TEXT_SYSTEM_PROMPT = f"""You are an intelligent and logical journalism scholar conducting analysis of news \
articles. Your task is to read the article and answer the following question about the article. Only output \
the json and no other text.

Framing is a way of classifying and categorizing information that allows audiences to make sense of and give \
meaning to the world around them (Goffman, 1974).
Entman (1993) has defined framing as "making some aspects of reality more salient in a text in order to \
promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment \
recommendation for the item described".
Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, \
images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).

A set of generic news frames with a name and description are:
{TEXT_FRAME_LIST_BLOCK}

Given the list of news frames, and the news article, carefully analyse the article and choose the appropriate \
frames used in the article from the above list. Only choose frames from the provided list. If none of the \
frames apply, choose "None" as the answer.

Respond with ONLY a JSON object (no markdown fences, no extra text; add escape characters where necessary to \
make it a valid JSON output):
{{"frames-list": ["<all frame names that apply from the list above>"], "reason": "<reasoning for the frames chosen>"}}"""

IMAGE_SYSTEM_PROMPT = f"""You are an intelligent and logical journalism scholar conducting analysis of images \
associated with news articles.

A set of generic news frames with a name and description are:
{IMAGE_FRAME_LIST_BLOCK}

You are given the full article title and text (for context — who/what it's about, what's happening) and its \
lead image. Carefully analyse the IMAGE and choose the appropriate frames from the above list based on what \
the image itself visually conveys, not the article's topic in the abstract. Only choose frames from the \
provided list. If none of the frames apply, choose "None" as the answer.

Respond with ONLY a JSON object (no markdown fences, no extra text; add escape characters where necessary to \
make it a valid JSON output):
{{"frames-list": ["<all frame names that apply from the list above>"], "reason": "<reasoning for the frames chosen>"}}"""


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
            names = parsed.get("frames-list", [])
            keys, unrecognized = [], []
            for name in names:
                key = CANONICAL_LABEL_TO_KEY.get(str(name).strip().lower())
                if key is None:
                    unrecognized.append(name)
                else:
                    keys.append(key)
            return {
                "frames": strip_none_key(keys),
                "unrecognized_frame_names": unrecognized,
                "explanation": parsed.get("reason", ""),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))
    return {"frames": [], "unrecognized_frame_names": [], "explanation": None,
            "error": f"relabel_failed_after_retries: {last_error}"}


def build_text_content(row):
    article_text = (row.get("article_text") or "")[:MAX_ARTICLE_CHARS]
    return f"TITLE: {row.get('title', '')}\n\nTEXT:\n{article_text}"


def relabel_text_one(client, model, row):
    text_content = build_text_content(row)
    text_result = call_llm(client, model, TEXT_SYSTEM_PROMPT, text_content)
    return {
        "uuid": row["uuid"],
        "new_text_generic_frame": text_result["frames"],
        "new_text_generic_frame_exp": text_result["explanation"],
        "new_text_generic_frame_error": text_result["error"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if not NEWS_SAMPLE_PATH.exists():
        raise SystemExit(f"{NEWS_SAMPLE_PATH} not found — run scripts/build_human_sample.py first")
    rows = read_jsonl(NEWS_SAMPLE_PATH)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out_path = relabel_model_path(args.model)

    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(relabel_text_one, client, args.model, row): row["uuid"] for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Relabeling text with {args.model}"):
            res = future.result()
            results.append(res)
            if res["new_text_generic_frame_error"]:
                errors += 1

    write_jsonl(out_path, results)
    print(f"\nRelabeled {len(results)} rows with {args.model} ({errors} had a call error after retries).")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
