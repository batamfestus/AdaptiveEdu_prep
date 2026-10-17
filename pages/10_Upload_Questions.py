"""
Admin page: upload questions from a CSV file, with optional diagram files.

Flow:  template -> upload CSV (+ diagrams) -> validation report -> preview -> save for review.
All the real work is in core/question_import.py; this page only draws it.
"""
import streamlit as st

from core.db import get_session
from core.images import ImageInfo
from core.question_import import (
    MAX_IMAGE_FILES, MAX_ROWS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, ImportFileError,
    errors_csv, parse_csv, save_valid_rows, template_csv, validate_rows,
)
from ui.components import render_question
from ui.session import init_page, require_admin

init_page("Upload Questions", icon="📤")
admin = require_admin()

st.title("📤 Upload questions")

# A message left by the previous run (we rerun the page after saving).
flash = st.session_state.pop("upload_flash", None)
if flash:
    st.success(flash)

st.markdown(
    "**Step 1.** Download the template and fill it in (Excel is fine: save as **CSV UTF-8**). "
    "Delete the two example rows first, or they will be treated as real questions."
)
st.download_button("⬇️ Download CSV template", data=template_csv(),
                   file_name="adaptprep_question_template.csv", mime="text/csv")

with st.expander("Column guide"):
    st.markdown(
        f"**Required:** `{'`, `'.join(REQUIRED_COLUMNS)}`  \n"
        f"**Optional:** `{'`, `'.join(OPTIONAL_COLUMNS)}`  \n\n"
        "- `topic` and `competency` must match the syllabus exactly (capitals don't matter).\n"
        "- `correct_option` is A, B, C or D. `difficulty` is 1 (easy) to 5 (hard).\n"
        "- `source` is `original`, `past_paper_inspired` or `ai_drafted`. Be honest: it is shown to reviewers.\n"
        "- **Diagrams:** put the file name in `image_file` (for example `circuit1.png`) and describe the "
        "diagram in `image_alt`. Then upload the image files in Step 2. Names are matched ignoring capitals.\n"
        "- A CSV can attach one diagram to the question itself. For diagrams in the options, use **Add Question**.\n"
        "- Write formulas between dollar signs, like `$v = u + at$`. For a price, write `\\$5`.\n"
        f"- Up to {MAX_ROWS} questions and {MAX_IMAGE_FILES} image files per upload."
    )

# Changing the key resets the uploaders after a successful save.
n = st.session_state.get("upload_n", 0)
file = st.file_uploader("**Step 2.** Upload your CSV", type=["csv"], key=f"csv_upload_{n}")
image_files = st.file_uploader(
    "Diagram images referenced by the CSV (optional)", type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True, key=f"img_upload_{n}",
)
if file is None:
    st.stop()

# ---- parse + validate (read-only: nothing is saved yet) ----
try:
    parsed = parse_csv(file.getvalue())
except ImportFileError as e:
    st.error(str(e))
    st.stop()

if parsed.ignored_columns:
    st.warning("Ignored unrecognised column(s): " + ", ".join(parsed.ignored_columns)
               + ". Check for typos in the header row.")

images = {f.name: f.getvalue() for f in image_files}
try:
    with get_session() as db:
        report = validate_rows(db, parsed, images)
except ImportFileError as e:
    st.error(str(e))
    st.stop()

if report.unused_images:
    st.warning("These uploaded images are not used by any row: " + ", ".join(report.unused_images))

n_ok, n_bad = len(report.ok_rows), len(report.error_rows)
n_warn = sum(1 for r in report.ok_rows if r.warnings)
n_img = sum(1 for r in report.ok_rows if r.image is not None)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows in file", len(report.results))
c2.metric("Ready to save", n_ok)
c3.metric("Need fixing", n_bad)
c4.metric("With diagrams", n_img)
c5.metric("Warnings", n_warn)

# ---- problems ----
if n_bad:
    st.subheader("❌ Rows that need fixing")
    st.dataframe([{"Row": r.row_number, "Problem": "; ".join(r.errors)} for r in report.error_rows],
                 hide_index=True)
    st.download_button("⬇️ Download these rows to fix and re-upload", data=errors_csv(report),
                       file_name="rows_to_fix.csv", mime="text/csv")

# ---- preview ----
if n_ok:
    st.subheader("👀 Preview")
    shown = report.ok_rows[:20]
    for r in shown:
        q = r.question
        previews = []
        if r.image is not None:  # the cleaned image carries its own bytes, since it isn't saved yet
            previews.append(ImageInfo(None, "stem", r.image_alt, r.image.width, r.image.height,
                                      r.image.sha256, data=r.image.data))
        with st.expander(f"Row {r.row_number} · {q.topic} › {q.competency} · difficulty {q.difficulty}"):
            render_question(q.text, q.options, q.correct_option, q.explanation, previews)
            st.caption(f"Source: {q.source}")
            for w in r.warnings:
                st.caption(f"⚠️ {w}")
    if n_ok > len(shown):
        st.caption(f"Showing the first {len(shown)} of {n_ok} valid rows.")

    st.markdown("**Step 3.** Happy with the preview? Valid rows go to the review queue (not live yet).")
    if st.button(f"Save {n_ok} question(s) for review", type="primary"):
        with get_session() as db:
            saved = save_valid_rows(db, report, admin, file.name)
        st.session_state["upload_flash"] = (
            f"Saved {saved} question(s) to the review queue."
            + (f" {n_bad} row(s) still need fixing." if n_bad else "")
        )
        st.session_state["upload_n"] = n + 1
        st.rerun()  # redraw with empty uploaders
else:
    st.info("No valid rows to save yet. Fix the problems above and upload again.")
