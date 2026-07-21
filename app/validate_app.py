"""
Streamlit interface for manually validating the consolidated mm-framing rows:
shows the real scraped article text + image, the dataset's ORIGINAL text/image
frame labels, the 2-of-3 majority-vote CONSOLIDATED labels from a 3-model
ensemble, and each individual model's own labels (expandable), prioritizing
rows the automated sweep (scripts/flag_issues.py) flagged.

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
    AGREEMENT_THRESHOLD,
    CANONICAL_FRAMES,
    CONSOLIDATED_PATH,
    DATA_DIR,
    ENSEMBLE_MODELS,
    FLAGS_PATH,
    VALIDATION_RESULTS_PATH,
    model_slug,
    normalize_frame_tags,
    parse_list_field,
    read_jsonl,
)

st.set_page_config(page_title="mm-framing validation", layout="wide")

VERDICT_OPTIONS = [
    "Not reviewed",
    "Consolidated labels look correct",
    "Consolidated text label wrong",
    "Consolidated image label wrong",
    "Both consolidated labels wrong",
    "Prefer the original labels",
    "Unclear / need more context",
]
CANONICAL_LABELS = {k: v.split(" — ")[0] for k, v in CANONICAL_FRAMES.items()}
MODEL_SLUGS = [model_slug(m) for m in ENSEMBLE_MODELS]


@st.cache_data
def load_data():
    rows = read_jsonl(CONSOLIDATED_PATH)
    flags = json.loads(FLAGS_PATH.read_text()) if FLAGS_PATH.exists() else {}
    return rows, flags, CONSOLIDATED_PATH.name


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


def old_frame_chip_line(raw_value):
    tags, _ = parse_list_field(raw_value)
    canonical, unknown = normalize_frame_tags(tags)
    parts = [f"`{CANONICAL_LABELS.get(c, c)}`" for c in canonical]
    parts += [f":red[`{t} (unknown)`]" for t in unknown]
    return " ".join(parts) if parts else "_(none)_"


def frame_chip_line(keys, strengths=None, votes=None):
    if not keys:
        return "_(none — no frame applies)_"
    parts = []
    for k in keys:
        label = CANONICAL_LABELS.get(k, k)
        extras = []
        if strengths and strengths.get(k):
            extras.append(strengths[k])
        if votes is not None:
            extras.append(f"{votes.get(k, 0)}/{len(ENSEMBLE_MODELS)}")
        parts.append(f"`{label}` :gray[({', '.join(extras)})]" if extras else f"`{label}`")
    return "  ".join(parts)


rows, flags_by_uuid, source_name = load_data()
rows_by_uuid = {r["uuid"]: r for r in rows}
uuids_all = [r["uuid"] for r in rows]

if "results" not in st.session_state:
    st.session_state.results = load_results()
if "idx" not in st.session_state:
    st.session_state.idx = 0

st.sidebar.title("mm-framing validation")
st.sidebar.caption(f"Source: `{source_name}` · {len(rows)} rows (non-news filtered out)")
st.sidebar.caption(f"Ensemble: {', '.join(ENSEMBLE_MODELS)} · keep if ≥{AGREEMENT_THRESHOLD}/{len(ENSEMBLE_MODELS)} agree")

view_mode = st.sidebar.radio("Show", ["All rows", "Flagged only"], index=0)
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
by_model = row.get("by_model") or {}

st.title(row.get("title", "(no title)"))
meta_cols = st.columns(4)
meta_cols[0].markdown(f"**Source**  \n{row.get('source_domain', '—')}")
meta_cols[1].markdown(f"**Leaning**  \n{row.get('political_leaning', '—')}")
meta_cols[2].markdown(f"**Published**  \n{row.get('date_publish', '—')}")
if row.get("url"):
    meta_cols[3].link_button("Open original article", row["url"], width='stretch')

if row_flags:
    with st.container(border=True):
        st.markdown("**⚠️ Automated flags for this row:**")
        for f in row_flags:
            st.markdown(f"- `{f['code']}`" + (f" — {f['detail']}" if f.get("detail") else ""))

img_col, text_col = st.columns([1, 1.4])

with img_col:
    st.subheader("Image")
    local_path = row.get("image_local_path")
    full_path = DATA_DIR / local_path if local_path else None
    if local_path and full_path and full_path.exists():
        st.image(str(full_path), width='stretch')
    else:
        st.warning("Image unavailable.")

    st.markdown(f"**Original img-generic-frame:** {old_frame_chip_line(row.get('img-generic-frame'))}")
    st.caption(row.get("img-frame-exp", ""))
    st.markdown(
        f"**Consolidated ({AGREEMENT_THRESHOLD}/{len(ENSEMBLE_MODELS)}):** "
        f"{frame_chip_line(row.get('consolidated_img_generic_frame') or [], votes=row.get('consolidated_img_frame_votes'))}"
    )
    with st.expander("Per-model image labels"):
        for slug in MODEL_SLUGS:
            m = by_model.get(slug, {})
            st.markdown(f"**{slug}:** {frame_chip_line(m.get('img_frames') or [], strengths=m.get('img_frame_strengths'))}")
            st.caption(m.get("img_exp") or "_(no explanation)_")

with text_col:
    st.subheader("Text")
    article_text = row.get("article_text")
    if article_text:
        with st.expander(f"Scraped article text ({row.get('article_word_count', 0)} words)", expanded=True):
            st.write(article_text)
    else:
        st.warning("Article text unavailable.")

    st.markdown(f"**Original text-generic-frame:** {old_frame_chip_line(row.get('text-generic-frame'))}")
    st.caption(row.get("text-generic-frame-exp", ""))
    st.markdown(
        f"**Consolidated ({AGREEMENT_THRESHOLD}/{len(ENSEMBLE_MODELS)}):** "
        f"{frame_chip_line(row.get('consolidated_text_generic_frame') or [], votes=row.get('consolidated_text_frame_votes'))}"
    )
    with st.expander("Per-model text labels"):
        for slug in MODEL_SLUGS:
            m = by_model.get(slug, {})
            st.markdown(f"**{slug}:** {frame_chip_line(m.get('text_frames') or [], strengths=m.get('text_frame_strengths'))}")
            st.caption(m.get("text_exp") or "_(no explanation)_")

st.divider()
st.subheader("Your validation")

existing = st.session_state.results.get(uuid, {})
with st.form(key=f"form_{uuid}"):
    existing_verdict = existing.get("verdict", "Not reviewed")
    verdict = st.selectbox(
        "Verdict", VERDICT_OPTIONS,
        index=VERDICT_OPTIONS.index(existing_verdict) if existing_verdict in VERDICT_OPTIONS else 0,
    )
    corrected_text_frames = st.multiselect(
        "Final text frame(s), if different from consolidated",
        list(CANONICAL_LABELS.values()),
        default=[v for v in existing.get("corrected_text_frames", "").split(";") if v],
    )
    corrected_img_frames = st.multiselect(
        "Final image frame(s), if different from consolidated",
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
