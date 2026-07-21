"""
Streamlit interface for manually validating the sampled mm-framing rows,
prioritizing rows the automated sweep (scripts/flag_issues.py) flagged.

Run (on the remote machine):
    streamlit run app/validate_app.py

Then forward port 8501 to view it locally (see README.md).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import (  # noqa: E402
    CANONICAL_FRAMES,
    DATA_DIR,
    FLAGS_PATH,
    FRAME_JUDGMENTS_PATH,
    SAMPLE_PATH,
    VALIDATION_RESULTS_PATH,
    normalize_frame_tags,
    read_jsonl,
)

st.set_page_config(page_title="mm-framing validation", layout="wide")

VERDICT_OPTIONS = [
    "Not reviewed",
    "Labels look correct",
    "Text frame incorrect",
    "Image frame incorrect",
    "Both frames incorrect",
    "Unclear / need more context",
    "Skip — data issue, can't judge",
]
CANONICAL_LABELS = {k: v.split(" — ")[0] for k, v in CANONICAL_FRAMES.items()}


@st.cache_data
def load_data():
    rows = read_jsonl(SAMPLE_PATH)
    flags = json.loads(FLAGS_PATH.read_text()) if FLAGS_PATH.exists() else {}
    judgments = json.loads(FRAME_JUDGMENTS_PATH.read_text()) if FRAME_JUDGMENTS_PATH.exists() else {}
    return rows, flags, judgments, SAMPLE_PATH.name


def load_results():
    if VALIDATION_RESULTS_PATH.exists():
        df = pd.read_csv(VALIDATION_RESULTS_PATH, dtype=str).fillna("")
        return {r["uuid"]: r.to_dict() for _, r in df.iterrows()}
    return {}


def save_result(uuid, record):
    results = st.session_state.results
    results[uuid] = record
    df = pd.DataFrame(results.values())
    VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(VALIDATION_RESULTS_PATH, index=False)


def frame_chip_line(tags):
    canonical, unknown = normalize_frame_tags(tags)
    parts = [f"`{CANONICAL_LABELS.get(c, c)}`" for c in canonical]
    parts += [f":red[`{t} (unknown)`]" for t in unknown]
    return " ".join(parts) if parts else "_(none)_"


def judge_verdict_block(verdict, reasoning):
    if verdict == "questionable":
        st.error(f"⚠️ **Judge: questionable** — {reasoning}")
    elif verdict == "accurate":
        st.success(f"✅ Judge: accurate — {reasoning}")
    else:
        st.caption("No automated judgment available for this row (run scripts/judge_frames.py).")


rows, flags_by_uuid, judgments_by_uuid, source_name = load_data()
rows_by_uuid = {r["uuid"]: r for r in rows}
uuids_all = [r["uuid"] for r in rows]

if "results" not in st.session_state:
    st.session_state.results = load_results()
if "idx" not in st.session_state:
    st.session_state.idx = 0

st.sidebar.title("mm-framing validation")
st.sidebar.caption(f"Source: `{source_name}` · {len(rows)} rows")

view_mode = st.sidebar.radio("Show", ["Flagged only", "All rows"], index=0)
if view_mode == "Flagged only":
    view_uuids = [u for u in uuids_all if flags_by_uuid.get(u)]
else:
    view_uuids = uuids_all

if not view_uuids:
    st.sidebar.success("No flagged rows — nothing to review in this view.")
    st.stop()

st.session_state.idx = min(st.session_state.idx, len(view_uuids) - 1)

reviewed = sum(
    1 for u in view_uuids
    if st.session_state.results.get(u, {}).get("verdict", "Not reviewed") != "Not reviewed"
)
st.sidebar.metric("Reviewed", f"{reviewed}/{len(view_uuids)}")

jump_options = [
    f"{'✅ ' if st.session_state.results.get(u, {}).get('verdict', 'Not reviewed') != 'Not reviewed' else '◻️ '}"
    f"{rows_by_uuid[u].get('title', '')[:60]}"
    for u in view_uuids
]
picked = st.sidebar.selectbox("Jump to row", jump_options, index=st.session_state.idx)
st.session_state.idx = jump_options.index(picked)

col_prev, col_next = st.sidebar.columns(2)
if col_prev.button("⬅ Prev", width='stretch') and st.session_state.idx > 0:
    st.session_state.idx -= 1
    st.rerun()
if col_next.button("Next ➡", width='stretch') and st.session_state.idx < len(view_uuids) - 1:
    st.session_state.idx += 1
    st.rerun()

uuid = view_uuids[st.session_state.idx]
row = rows_by_uuid[uuid]
row_flags = flags_by_uuid.get(uuid, [])
judgment = judgments_by_uuid.get(uuid, {})

st.title(row.get("title", "(no title)"))
meta_cols = st.columns(4)
meta_cols[0].markdown(f"**Source**  \n{row.get('source_domain', '—')}")
meta_cols[1].markdown(f"**Leaning**  \n{row.get('political_leaning', '—')}")
meta_cols[2].markdown(f"**Published**  \n{row.get('date_publish', '—')}")
if row.get("url"):
    meta_cols[3].link_button("Open original article", row["url"], width='stretch')

non_judgment_flags = [
    f for f in row_flags
    if f["code"] not in ("text_frame_questionable", "img_frame_questionable")
]
if non_judgment_flags:
    with st.container(border=True):
        st.markdown("**⚠️ Other automated flags for this row:**")
        for f in non_judgment_flags:
            st.markdown(f"- `{f['code']}`" + (f" — {f['detail']}" if f.get("detail") else ""))

img_col, text_col = st.columns([1, 1.4])

with img_col:
    st.subheader("Image")
    local_path = row.get("image_local_path")
    full_path = DATA_DIR / local_path if local_path else None
    if local_path and full_path and full_path.exists():
        st.image(str(full_path), width='stretch')
    else:
        st.warning(f"Image unavailable ({row.get('image_error', 'not scraped')}). Check the original article link above.")

    st.markdown(f"**img-generic-frame:** {frame_chip_line(row.get('_img_generic_frame_parsed') or [])}")
    st.caption(f"Original model justification: {row.get('img-frame-exp', '')}")
    judge_verdict_block(judgment.get("img_frame_verdict"), judgment.get("img_frame_reasoning"))
    st.markdown(f"**img-entity-name:** {row.get('img-entity-name', '—')}  ·  **sentiment:** {row.get('img-entity-sentiment', '—')}")
    st.caption(row.get("img-entity-sentiment-exp", ""))

with text_col:
    st.subheader("Text")
    article_text = row.get("article_text")
    if article_text:
        with st.expander(f"Scraped article text ({row.get('article_word_count', 0)} words)", expanded=True):
            st.write(article_text)
    else:
        st.warning(f"Article text unavailable ({row.get('scrape_error', 'not scraped')}). Check the original article link above.")

    st.markdown(f"**topic:** {row.get('text-topic', '—')}")
    st.caption(row.get("text-topic-exp", ""))
    st.markdown(f"**text-generic-frame:** {frame_chip_line(row.get('_text_generic_frame_parsed') or [])}")
    st.caption(f"Original model justification: {row.get('text-generic-frame-exp', '')}")
    judge_verdict_block(judgment.get("text_frame_verdict"), judgment.get("text_frame_reasoning"))
    st.markdown(f"**issue-frame:** {row.get('text-issue-frame', '—')}")
    st.caption(row.get("text-issue-frame-exp", ""))
    st.markdown(f"**entity:** {row.get('text-entity-name', '—')}  ·  **sentiment:** {row.get('text-entity-sentiment', '—')}")
    st.caption(row.get("text-entity-sentiment-exp", ""))

st.divider()
st.subheader("Your validation")

existing = st.session_state.results.get(uuid, {})
with st.form(key=f"form_{uuid}"):
    verdict = st.selectbox(
        "Verdict", VERDICT_OPTIONS,
        index=VERDICT_OPTIONS.index(existing.get("verdict", "Not reviewed")),
    )
    corrected_text_frames = st.multiselect(
        "If the text frame is wrong — what should it be?",
        list(CANONICAL_LABELS.values()),
        default=[v for v in existing.get("corrected_text_frames", "").split(";") if v],
    )
    corrected_img_frames = st.multiselect(
        "If the image frame is wrong — what should it be?",
        list(CANONICAL_LABELS.values()),
        default=[v for v in existing.get("corrected_img_frames", "").split(";") if v],
    )
    notes = st.text_area("Notes", value=existing.get("notes", ""))
    submitted = st.form_submit_button("Save", width='stretch')
    if submitted:
        save_result(uuid, {
            "uuid": uuid,
            "title": row.get("title", ""),
            "verdict": verdict,
            "corrected_text_frames": ";".join(corrected_text_frames),
            "corrected_img_frames": ";".join(corrected_img_frames),
            "notes": notes,
        })
        st.success("Saved.")
        st.rerun()
