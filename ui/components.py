"""
ui/components.py
----------------
Small reusable pieces of Streamlit UI shared by several pages.
"""
import streamlit as st

from core.images import ImageInfo


@st.cache_data(show_spinner=False, max_entries=256)
def _load_image(image_id: int, sha256: str) -> bytes:
    """Fetch a saved image's bytes from the database ONCE, then reuse them.

    Caching matters: Streamlit reruns the page on every click, and re-downloading
    every diagram from the database each time would be slow.
    `sha256` is part of the cache key on purpose: if an image is ever replaced,
    its hash changes, so we can never show a stale picture.
    """
    from core.db import get_session
    from core.images import get_image_bytes

    with get_session() as db:
        return get_image_bytes(db, image_id)


def image_bytes(image: ImageInfo) -> bytes:
    """The picture's bytes. Saved images are loaded by id; previews carry their own bytes."""
    return image.data if image.data is not None else _load_image(image.id, image.sha256)


def show_image(image: ImageInfo, show_caption: bool = True, caption: str | None = None) -> None:
    """Display one diagram, with its description (or a caption of your choice) underneath."""
    st.image(image_bytes(image), caption=caption or (image.alt_text if show_caption else None))


def render_question(text: str, options: dict, correct: str | None = None,
                    explanation: str = "", images: list[ImageInfo] | None = None,
                    show_image_captions: bool = True) -> None:
    """Show a question the way a student will see it (admin view marks the answer).

    `images` may hold a "stem" diagram (shown under the question text) and
    diagrams for individual options (shown under that option).

    SECURITY NOTE: we use st.markdown WITHOUT unsafe_allow_html. Uploaded text
    is untrusted, so it must never be allowed to inject HTML or scripts.
    Formulas can be written between dollar signs, e.g. $v = u + at$.
    """
    by_role = {img.role: img for img in (images or [])}

    st.markdown(text)
    if "stem" in by_role:
        show_image(by_role["stem"], show_image_captions)

    for key in sorted(options):
        marker = "✅" if key == correct else "▫️"
        st.markdown(f"{marker} **{key}.** {options[key]}")
        if key in by_role:
            show_image(by_role[key], show_image_captions)

    if explanation:
        st.info(f"**Explanation:** {explanation}")
    elif correct is not None:
        st.warning("No explanation yet.")


def render_review_item(item) -> None:
    """Show one graded question: what the student chose, what was right, and why.

    `item` is a core.cbt.ResultItem. This only ever runs AFTER a test is submitted,
    because that is the only time core/cbt.py hands out the answer key.
    (Lesson 6 adds an "Explain with AI" button right here.)
    """
    by_role = {img.role: img for img in item.images}
    st.markdown(item.text)
    if "stem" in by_role:
        show_image(by_role["stem"])

    for key in sorted(item.options):
        if key == item.correct_option:
            marker = "✅"
        elif key == item.selected:
            marker = "❌"
        else:
            marker = "▫️"
        suffix = "  ← your answer" if key == item.selected else ""
        st.markdown(f"{marker} **{key}.** {item.options[key]}{suffix}")
        if key in by_role:
            show_image(by_role[key])

    if item.selected is None:
        st.warning("You did not answer this question.")
    if item.explanation:
        st.info(f"**Explanation:** {item.explanation}")
    else:
        st.caption("No explanation is available for this question yet.")
