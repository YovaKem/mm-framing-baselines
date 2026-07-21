"""
Filter out rows that aren't genuine news/journalism (game announcements,
"best restaurants in X" listicles, lifestyle round-ups, etc.) — framing
analysis assumes editorial choices about how a real-world event/issue is
covered, which doesn't really apply to that kind of content.

Uses a cheap text-only LLM call per row (article title + scraped text) via
OpenRouter. Requires OPENROUTER_API_KEY in .env.

Run after build_sample.py.

Usage:
    python scripts/filter_news.py
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from common import (
    NEWS_FILTER_PATH,
    NEWS_SAMPLE_PATH,
    OPENROUTER_MODEL,
    SAMPLE_PATH,
    extract_json_object,
    read_jsonl,
    write_jsonl,
)

load_dotenv()

MAX_ARTICLE_CHARS = 3000

SYSTEM_PROMPT = """You are doing quality control on a news-framing-analysis dataset. Framing \
analysis studies how journalists select which aspects of a real-world event, issue, or \
controversy to emphasize — it only makes sense for genuine news/journalism content.

Given an article's title and text, decide if it IS genuine news/journalism (reporting on \
real-world events: politics, policy, crime, business, world affairs, science, disasters, \
court cases, etc. — opinion/analysis pieces about real events count too) or is NOT \
(entertainment/gaming product announcements and reviews, "best of" travel/dining listicles, \
lifestyle round-ups, horoscopes, quizzes, recipe posts, or similar content with no real \
editorial framing of a real-world issue).

Respond with ONLY a JSON object (no markdown fences, no extra text) with exactly these keys: \
is_news (true or false), reason (one sentence)."""


def classify_one(client, model, row, retries=2):
    uuid = row["uuid"]
    user_text = f"TITLE: {row.get('title', '')}\n\nTEXT:\n{(row.get('article_text') or '')[:MAX_ARTICLE_CHARS]}"
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
            )
            parsed = extract_json_object(resp.choices[0].message.content)
            return {"uuid": uuid, "is_news": bool(parsed["is_news"]), "reason": parsed.get("reason", ""), "error": None}
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))
    # Fail open: an unclassifiable row is kept (marked is_news=True) rather than silently dropped.
    return {"uuid": uuid, "is_news": True, "reason": None, "error": f"classify_failed_after_retries: {last_error}"}


def main():
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found — run scripts/build_sample.py first")
    rows = read_jsonl(SAMPLE_PATH)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(classify_one, client, OPENROUTER_MODEL, row): row["uuid"] for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering non-news"):
            res = future.result()
            results[res["uuid"]] = res

    NEWS_FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NEWS_FILTER_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    kept = [row for row in rows if results[row["uuid"]]["is_news"]]
    removed = [row for row in rows if not results[row["uuid"]]["is_news"]]
    write_jsonl(NEWS_SAMPLE_PATH, kept)

    print(f"\nKept {len(kept)}/{len(rows)} rows as genuine news.")
    print(f"Removed {len(removed)} non-news rows:")
    for row in removed:
        print(f"  - {row['title'][:70]!r} — {results[row['uuid']]['reason']}")
    print(f"\nWrote {NEWS_SAMPLE_PATH} ({len(kept)} rows) and {NEWS_FILTER_PATH}")


if __name__ == "__main__":
    main()
