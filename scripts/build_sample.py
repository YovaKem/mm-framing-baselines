"""
Build a 300-row sample where EVERY row has both article text and a lead image
successfully scraped from the original `url` (the dataset itself stores
neither — see docs/DATASET.md).

Because live 2023-2024 news links have real rot (paywalls, dead pages, no
og:image), a plain one-shot random sample of 300 loses roughly half its rows.
Instead this script draws from a fixed, seeded shuffle of the
valid_framing_subset split and keeps pulling further into that shuffled order
until 300 rows clear both bars, logging every attempt (kept or rejected, and
why) to data/scrape_attempts_log.jsonl for transparency.

IMPORTANT CAVEAT this introduces: the final 300 rows are a random sample
*conditioned on being scrapeable* — i.e. biased toward outlets/links that are
still live and unpaywalled — not a strict random sample of the full dataset.
See docs/DATASET.md.

Usage:
    python scripts/build_sample.py
    python scripts/build_sample.py --n 300 --seed 42 --workers 12 --max-attempts 2000
"""
import argparse
import mimetypes
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
import trafilatura
from bs4 import BeautifulSoup
from datasets import load_dataset
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from common import (
    HF_DATASET,
    HF_SPLIT,
    IMAGES_DIR,
    MIN_ARTICLE_WORDS,
    RANDOM_SEED,
    SAMPLE_PATH,
    SAMPLE_SIZE,
    SCRAPE_ATTEMPTS_LOG_PATH,
    parse_list_field,
    write_jsonl,
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
BATCH_SIZE = 40


def make_session():
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return session


def find_lead_image_url(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    for attr, val in [("property", "og:image"), ("name", "twitter:image"), ("property", "og:image:url")]:
        tag = soup.find("meta", attrs={attr: val})
        if tag and tag.get("content"):
            return urljoin(base_url, tag["content"].strip())
    article = soup.find("article")
    if article:
        img = article.find("img")
        if img and img.get("src"):
            return urljoin(base_url, img["src"].strip())
    return None


def download_image(session, image_url, uuid, timeout):
    try:
        resp = session.get(image_url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return None, f"not_an_image_content_type: {content_type}"
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
        if ext == ".jpe":
            ext = ".jpg"
        out_path = IMAGES_DIR / f"{uuid}{ext}"
        size = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > MAX_IMAGE_BYTES:
                    f.close()
                    out_path.unlink(missing_ok=True)
                    return None, "image_too_large"
                f.write(chunk)
        return str(out_path.relative_to(IMAGES_DIR.parent)), None
    except requests.RequestException as e:
        return None, f"image_download_error: {e}"


def scrape_one(session, row, timeout):
    uuid = row["uuid"]
    url = row.get("url")
    result = dict(
        uuid=uuid, scrape_http_status=None, scrape_error=None, article_text=None,
        article_word_count=0, image_local_path=None, image_error=None,
    )
    if not url:
        result["scrape_error"] = "no_url_in_row"
        return result

    time.sleep(random.uniform(0.15, 0.45))
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        result["scrape_http_status"] = resp.status_code
        resp.raise_for_status()
    except requests.RequestException as e:
        result["scrape_error"] = f"fetch_error: {e}"
        return result

    html, final_url = resp.text, resp.url

    extracted = trafilatura.extract(html, url=final_url, favor_precision=True)
    if extracted:
        result["article_text"] = extracted
        result["article_word_count"] = len(extracted.split())
    else:
        result["scrape_error"] = "trafilatura_extraction_failed"

    image_url = find_lead_image_url(html, final_url)
    if image_url:
        local_path, image_err = download_image(session, image_url, uuid, timeout)
        result["image_local_path"] = local_path
        result["image_error"] = image_err
    else:
        result["image_error"] = "no_og_image_found"

    return result


def is_complete(result):
    return (
        bool(result.get("article_text"))
        and result.get("article_word_count", 0) >= MIN_ARTICLE_WORDS
        and bool(result.get("image_local_path"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=2000)
    args = parser.parse_args()

    print(f"Loading {HF_DATASET} split={HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    shuffled = ds.shuffle(seed=args.seed)
    print(f"Loaded {len(shuffled):,} rows, shuffled with seed={args.seed}.")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    accepted, attempts_log = [], []
    cursor = 0
    pbar = tqdm(total=args.n, desc="Complete samples collected")

    while len(accepted) < args.n and cursor < len(shuffled) and len(attempts_log) < args.max_attempts:
        end = min(cursor + BATCH_SIZE, len(shuffled))
        batch = shuffled.select(range(cursor, end))
        cursor = end

        rows = list(batch)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(scrape_one, session, row, args.timeout): row for row in rows}
            for future in as_completed(futures):
                row = futures[future]
                res = future.result()
                attempts_log.append({**res, "kept": is_complete(res)})
                if is_complete(res) and len(accepted) < args.n:
                    text_tags, text_err = parse_list_field(row.get("text-generic-frame"))
                    img_tags, img_err = parse_list_field(row.get("img-generic-frame"))
                    merged = dict(row)
                    merged.update(res)
                    merged["_text_generic_frame_parsed"] = text_tags
                    merged["_text_generic_frame_parse_error"] = text_err
                    merged["_img_generic_frame_parsed"] = img_tags
                    merged["_img_generic_frame_parse_error"] = img_err
                    accepted.append(merged)
                    pbar.update(1)

        pbar.set_postfix(attempted=len(attempts_log), accept_rate=f"{len(accepted) / len(attempts_log):.0%}")

    pbar.close()

    write_jsonl(SAMPLE_PATH, accepted[: args.n])
    write_jsonl(SCRAPE_ATTEMPTS_LOG_PATH, attempts_log)

    print(f"\nCollected {len(accepted)}/{args.n} complete rows after {len(attempts_log)} attempts "
          f"({len(accepted) / len(attempts_log):.0%} acceptance rate).")
    if len(accepted) < args.n:
        print(f"WARNING: ran out of rows/attempts before reaching {args.n}. "
              f"Re-run with a higher --max-attempts or fall back to the 'full' split.")
    print(f"Wrote sample to {SAMPLE_PATH}")
    print(f"Wrote full attempt log (incl. rejected rows) to {SCRAPE_ATTEMPTS_LOG_PATH}")


if __name__ == "__main__":
    main()
