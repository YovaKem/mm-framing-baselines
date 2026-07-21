"""
Best-effort scrape of the original article text + lead image for each row in
sample_300.jsonl. The dataset itself stores neither (see docs/DATASET.md) —
only the LLM's derived labels/explanations and the source `url`. This script
fetches the live page once, caches what it finds locally, and records why a
row failed when it does (dead link, paywall, no og:image, etc.) so those
failures can be surfaced as review flags rather than silently missing data.

Usage:
    python scripts/scrape_articles.py
    python scripts/scrape_articles.py --workers 8 --timeout 15
"""
import argparse
import mimetypes
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from common import IMAGES_DIR, SAMPLE_PATH, SCRAPED_PATH, read_jsonl, write_jsonl

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


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
    article_img = soup.find("article")
    if article_img:
        img = article_img.find("img")
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
    result = {
        "uuid": uuid,
        "scrape_http_status": None,
        "scrape_error": None,
        "article_text": None,
        "article_word_count": 0,
        "image_local_path": None,
        "image_error": None,
    }
    if not url:
        result["scrape_error"] = "no_url_in_row"
        return result

    time.sleep(random.uniform(0.2, 0.6))
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        result["scrape_http_status"] = resp.status_code
        resp.raise_for_status()
    except requests.RequestException as e:
        result["scrape_error"] = f"fetch_error: {e}"
        return result

    html = resp.text
    final_url = resp.url

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    if not SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found — run scripts/sample_dataset.py first")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(SAMPLE_PATH)
    session = make_session()

    results_by_uuid = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scrape_one, session, row, args.timeout): row["uuid"] for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping"):
            res = future.result()
            results_by_uuid[res["uuid"]] = res

    merged = []
    ok, text_failed, image_failed = 0, 0, 0
    for row in rows:
        res = results_by_uuid.get(row["uuid"], {})
        row.update(res)
        merged.append(row)
        if res.get("article_text"):
            ok += 1
        else:
            text_failed += 1
        if not res.get("image_local_path"):
            image_failed += 1

    write_jsonl(SCRAPED_PATH, merged)
    print(f"\nWrote {len(merged)} rows to {SCRAPED_PATH}")
    print(f"  article text recovered: {ok}/{len(merged)}  (failed: {text_failed})")
    print(f"  images recovered: {len(merged) - image_failed}/{len(merged)}  (failed: {image_failed})")


if __name__ == "__main__":
    main()
