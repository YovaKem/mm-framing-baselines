"""
Build the current sample: the 578 articles from data/human_image_frame_labels.csv
(the paper's own human double-annotation pass over images, arXiv:2503.20960) that
have a matching row — url, title, and the dataset's original metadata/labels — in
the full copenlu/mm-framing HF dataset (`full`, ~479k rows; NOT the smaller
`valid_framing_subset` the old pipeline sampled from, which only covers a third of
these uuids). 22 of the CSV's 600 uuids have no match anywhere in the dataset and
are dropped (logged, not silently discarded).

Unlike build_sample.py, this script does NOT scrape live images — the human
annotators labeled one specific image per article, so the image comes from the
zip extracted into data/images/ (uuid-keyed) ahead of time, not from whatever a
live page happens to serve today. Only article TEXT is scraped live (the CSV/HF
dataset store no article text either).

Also assigns a fixed split via a seeded shuffle: SPLIT_TEST_SIZE rows are held
out as the shared eval set for every baseline (zero-shot and LoRA-finetuned
alike); the rest are training data for the two LoRA finetunes only.

No news/genuine-journalism filtering is applied to this sample (unlike the old
238-article one) — this script's output is the final sample every downstream
script reads.

Usage:
    python scripts/build_human_sample.py
    python scripts/build_human_sample.py --workers 12 --limit 20   # smoke test
"""
import argparse
import ast
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import trafilatura
from datasets import load_dataset
from tqdm import tqdm

from build_sample import make_session
from common import (
    HF_DATASET,
    HF_FULL_DATA_FILE,
    HUMAN_LABELS_CSV_PATH,
    IMAGES_DIR,
    MIN_ARTICLE_WORDS,
    NEWS_SAMPLE_PATH,
    RANDOM_SEED,
    SCRAPE_ATTEMPTS_LOG_PATH,
    SPLIT_TEST_SIZE,
    SPLIT_TRAIN_LORA_SIZE,
    normalize_frame_tags,
    strip_none_key,
    write_jsonl,
)

MAX_UUID_LOG = 30


def load_human_labels():
    import csv

    rows = []
    with open(HUMAN_LABELS_CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                raw_tags = ast.literal_eval(row["merged_human_frames"])
            except (ValueError, SyntaxError):
                raw_tags = []
            raw_tags = [str(t).strip().lower() for t in raw_tags]
            canonical, unknown = normalize_frame_tags(raw_tags)
            canonical = strip_none_key(canonical)
            rows.append({
                "uuid": row["uuid"],
                "merged_human_frames_raw": row["merged_human_frames"],
                "human_img_generic_frame": canonical,
                "human_img_generic_frame_unknown_tags": unknown,
            })
    return rows


def assign_splits(uuids, seed):
    shuffled = list(uuids)
    random.Random(seed).shuffle(shuffled)
    test_uuids = set(shuffled[:SPLIT_TEST_SIZE])
    train_uuids = set(shuffled[SPLIT_TEST_SIZE:SPLIT_TEST_SIZE + SPLIT_TRAIN_LORA_SIZE])
    return {u: ("test" if u in test_uuids else "train_lora" if u in train_uuids else None) for u in uuids}


def extract_result(uuid, html, final_url, source):
    result = {"uuid": uuid, "article_text": None, "article_word_count": 0,
              "scrape_error": None, "scrape_source": source}
    extracted = trafilatura.extract(html, url=final_url, favor_precision=True) if html else None
    if extracted:
        result["article_text"] = extracted
        result["article_word_count"] = len(extracted.split())
    else:
        result["scrape_error"] = "trafilatura_extraction_failed"
    return result


def fetch_live_one(session, uuid, url, timeout):
    result = {"uuid": uuid, "scrape_http_status": None, "scrape_error": None,
              "article_text": None, "article_word_count": 0, "scrape_source": None}
    if not url:
        result["scrape_error"] = "no_url_in_row"
        return result

    time.sleep(random.uniform(0.15, 0.45))
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        result["scrape_http_status"] = resp.status_code
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        result["scrape_error"] = f"fetch_error: {e}"
        return result

    result.update(extract_result(uuid, resp.text, resp.url, "live"))
    return result


def fetch_wayback_one(session, uuid, url, date_publish, timeout, retries=3):
    """Fallback for live-fetch failures (bot walls, dead links): pull the closest
    archived snapshot from the Wayback Machine instead. A legitimate, standard
    workaround for link rot — not a live-site bypass — since we're just reading a
    public archive of a page that used to be openly accessible. Run at LOW
    concurrency with backoff (see main()): archive.org's public API rate-limits
    (429) hard under bursts, so this deliberately isn't parallelized like the
    live-fetch pass."""
    result = {"uuid": uuid, "scrape_error": None, "article_text": None,
              "article_word_count": 0, "scrape_source": None}
    if not url:
        result["scrape_error"] = "no_url_in_row"
        return result

    ts = (date_publish or "").replace("-", "").replace(":", "").replace(" ", "")[:8] or None
    last_error = None
    for attempt in range(retries):
        try:
            avail = session.get(
                "http://archive.org/wayback/available",
                params={"url": url, **({"timestamp": ts} if ts else {})},
                timeout=timeout,
            )
            if avail.status_code == 429:
                last_error = "wayback_available_429"
                time.sleep(2 * (attempt + 1))
                continue
            avail.raise_for_status()
            snapshot = (avail.json().get("archived_snapshots") or {}).get("closest")
            if not snapshot or not snapshot.get("available"):
                result["scrape_error"] = "no_wayback_snapshot"
                return result
            resp = session.get(snapshot["url"], timeout=timeout)
            if resp.status_code == 429:
                last_error = "wayback_snapshot_429"
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            result.update(extract_result(uuid, resp.text, snapshot["url"], "wayback"))
            return result
        except Exception as e:  # noqa: BLE001
            last_error = f"wayback_error: {e}"
            time.sleep(1.5 * (attempt + 1))
    result["scrape_error"] = last_error or "wayback_failed_after_retries"
    return result


def image_path_for(uuid):
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = IMAGES_DIR / f"{uuid}{ext}"
        if p.exists():
            return str(p.relative_to(IMAGES_DIR.parent))
    return None


def is_complete(scrape_result, image_local_path):
    return (
        bool(scrape_result.get("article_text"))
        and scrape_result.get("article_word_count", 0) >= MIN_ARTICLE_WORDS
        and image_local_path is not None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--wayback-workers", type=int, default=3, help="kept low — archive.org rate-limits (429) hard under bursts")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--limit", type=int, default=None, help="only attempt the first N matched rows (smoke test)")
    args = parser.parse_args()

    human_rows = load_human_labels()
    print(f"Loaded {len(human_rows)} rows from {HUMAN_LABELS_CSV_PATH}")

    print(f"Loading {HF_DATASET} (full split, {HF_FULL_DATA_FILE}) ...")
    ds_full = load_dataset(HF_DATASET, data_files={"full": HF_FULL_DATA_FILE}, split="full")
    by_uuid = {u: i for i, u in enumerate(ds_full["uuid"])}
    print(f"Loaded {len(by_uuid):,} rows.")

    matched = [r for r in human_rows if r["uuid"] in by_uuid]
    unmatched = [r for r in human_rows if r["uuid"] not in by_uuid]
    print(f"\n{len(matched)}/{len(human_rows)} human-labeled uuids matched in the full HF dataset.")
    print(f"{len(unmatched)} uuids have no match anywhere and are dropped:")
    for r in unmatched[:MAX_UUID_LOG]:
        print(f"  - {r['uuid']}")
    if len(unmatched) > MAX_UUID_LOG:
        print(f"  ... and {len(unmatched) - MAX_UUID_LOG} more")

    if args.limit:
        matched = matched[: args.limit]

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # Phase 1: live fetch, high concurrency (independent hosts, safe to parallelize hard).
    live_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_live_one, session, r["uuid"], ds_full[by_uuid[r["uuid"]]]["url"], args.timeout): r
            for r in matched
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping article text (live)"):
            r = futures[future]
            live_results[r["uuid"]] = future.result()

    # Phase 2: Wayback fallback, LOW concurrency + backoff (a single shared archive.org
    # host, unlike phase 1's many independent news domains) — for anything phase 1
    # failed on outright, OR where it "succeeded" but under MIN_ARTICLE_WORDS (a
    # paywall/teaser snippet rather than the full article; the archived snapshot is
    # sometimes from before a paywall went up, or just a fuller capture).
    needs_wayback = [
        r for r in matched
        if live_results[r["uuid"]].get("scrape_error")
        or live_results[r["uuid"]].get("article_word_count", 0) < MIN_ARTICLE_WORDS
    ]
    wayback_results = {}
    if needs_wayback:
        with ThreadPoolExecutor(max_workers=args.wayback_workers) as pool:
            futures = {
                pool.submit(
                    fetch_wayback_one, session, r["uuid"], ds_full[by_uuid[r["uuid"]]]["url"],
                    ds_full[by_uuid[r["uuid"]]].get("date_publish"), args.timeout,
                ): r
                for r in needs_wayback
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping article text (wayback fallback)"):
                r = futures[future]
                wayback_results[r["uuid"]] = future.result()

    accepted, attempts_log = [], []
    for r in matched:
        ds_row = ds_full[by_uuid[r["uuid"]]]
        scrape_res = live_results[r["uuid"]]
        wb = wayback_results.get(r["uuid"])
        # Only take the wayback result if it's actually better (longer) than what live
        # gave us — a thin live snippet plus a thin/failed wayback attempt should still
        # report the live snippet's own error/word-count, not silently swap in a worse one.
        if wb and wb.get("article_word_count", 0) > scrape_res.get("article_word_count", 0):
            scrape_res = {**scrape_res, **wb, "scrape_error": wb.get("scrape_error"),
                          "live_scrape_error": scrape_res.get("scrape_error")}

        image_local_path = image_path_for(r["uuid"])
        complete = is_complete(scrape_res, image_local_path)
        attempts_log.append({**scrape_res, "image_local_path": image_local_path, "kept": complete})

        if complete:
            merged = dict(ds_row)
            merged.update(scrape_res)
            merged["image_local_path"] = image_local_path
            merged["human_img_generic_frame"] = r["human_img_generic_frame"]
            merged["human_img_generic_frame_raw"] = r["merged_human_frames_raw"]
            merged["human_img_generic_frame_unknown_tags"] = r["human_img_generic_frame_unknown_tags"]
            accepted.append(merged)

    # Assign splits AFTER scraping, from whatever actually succeeded — assigning up
    # front to a fixed-size subset of the 578 candidates (before knowing which would
    # scrape) meant scrape failures within that subset went unfilled by successes
    # among the uuids that weren't pre-assigned, undershooting the target sizes even
    # when enough total rows were actually available.
    split_by_uuid = assign_splits([r["uuid"] for r in accepted], args.seed)
    for r in accepted:
        r["split"] = split_by_uuid[r["uuid"]]
    accepted = [r for r in accepted if r["split"] is not None]

    write_jsonl(NEWS_SAMPLE_PATH, accepted)
    write_jsonl(SCRAPE_ATTEMPTS_LOG_PATH, attempts_log)

    n_test = sum(1 for r in accepted if r["split"] == "test")
    n_train = sum(1 for r in accepted if r["split"] == "train_lora")
    n_complete = sum(1 for a in attempts_log if a["kept"])
    print(f"\n{n_complete}/{len(matched)} matched uuids scraped+imaged successfully "
          f"({n_complete / len(matched):.0%}).")
    print(f"Assigned to splits: {len(accepted)}/{n_complete} "
          f"(test target {SPLIT_TEST_SIZE}, train_lora target {SPLIT_TRAIN_LORA_SIZE})")
    print(f"  split=test: {n_test} (target {min(SPLIT_TEST_SIZE, n_complete)})")
    print(f"  split=train_lora: {n_train} (target {min(SPLIT_TRAIN_LORA_SIZE, max(0, n_complete - SPLIT_TEST_SIZE))})")
    print(f"Wrote sample to {NEWS_SAMPLE_PATH}")
    print(f"Wrote attempt log (incl. rejected rows) to {SCRAPE_ATTEMPTS_LOG_PATH}")


if __name__ == "__main__":
    main()
