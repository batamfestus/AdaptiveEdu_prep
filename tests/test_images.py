"""
Tests for core/images.py. Run with:  pytest -q

An uploaded image is untrusted input, so most of these tests are attacks and
mistakes: fake files, huge files, hidden metadata, transparent backgrounds.
"""
import io

import pytest
from PIL import Image
from sqlalchemy import select

from core.images import (
    ImageError, process_image, remove_question_image, set_question_image,
    validate_alt_text, list_image_infos, get_image_bytes,
)
from core import images as images_module
from core.orm import Competency, Question, QuestionImage, QuestionReview, QuestionStatus as S
from core.permissions import PermissionDenied

ALT = "Ray diagram showing a ray bending towards the normal in glass."


def encode(img: Image.Image, fmt="PNG", **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def make_question(db, status=S.PENDING_REVIEW) -> Question:
    comp = db.scalars(select(Competency).where(Competency.name == "Apply Ohm's law")).one()
    q = Question(competency_id=comp.id, text="Q?", options={"A": "1", "B": "2", "C": "3", "D": "4"},
                 correct_option="A", difficulty=2, status=status)
    db.add(q)
    db.commit()
    return q


# ------------------------------------------------------------------ cleaning
def test_valid_png_is_accepted():
    out = process_image(encode(Image.new("RGB", (300, 200), "white")))
    assert out.content_type == "image/png" and (out.width, out.height) == (300, 200)
    assert len(out.sha256) == 64


def test_text_file_pretending_to_be_an_image_is_rejected():
    with pytest.raises(ImageError, match="not a valid image"):
        process_image(b"just some text, not a picture")


def test_svg_is_rejected():
    with pytest.raises(ImageError, match="not a valid image"):
        process_image(b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')


def test_gif_is_not_an_allowed_format():
    with pytest.raises(ImageError, match="GIF"):
        process_image(encode(Image.new("P", (10, 10)), "GIF"))


def test_empty_and_oversized_files_are_rejected(monkeypatch):
    with pytest.raises(ImageError, match="empty"):
        process_image(b"")
    monkeypatch.setattr(images_module, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ImageError, match="larger than"):
        process_image(encode(Image.new("RGB", (300, 300), "red")))


def test_pixel_bomb_is_rejected_before_decoding():
    # A 1-bit 6000 x 6000 image is only a few KB on disk but 36 million pixels.
    bomb = encode(Image.new("1", (6000, 6000)))
    assert len(bomb) < 100_000
    with pytest.raises(ImageError, match="too large"):
        process_image(bomb)


def test_truncated_file_is_reported_as_damaged():
    good = encode(Image.new("RGB", (400, 400), "blue"))
    with pytest.raises(ImageError):
        process_image(good[: len(good) // 2])


def test_big_images_are_shrunk():
    out = process_image(encode(Image.new("RGB", (3000, 1500), "white")))
    assert max(out.width, out.height) == 1200 and out.width == 2 * out.height


def test_small_images_are_not_enlarged():
    out = process_image(encode(Image.new("RGB", (100, 50), "white")))
    assert (out.width, out.height) == (100, 50)


def test_transparent_background_becomes_white():
    # A black line drawing on a TRANSPARENT background would vanish in dark mode.
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    out = process_image(encode(img))
    pixel = Image.open(io.BytesIO(out.data)).convert("RGB").getpixel((5, 5))
    assert pixel == (255, 255, 255)


def test_metadata_is_stripped():
    img = Image.new("RGB", (50, 50), "green")
    exif = Image.Exif()
    exif[0x010F] = "SecretCameraMaker"          # tag 'Make'
    raw = encode(img, "JPEG", exif=exif)
    assert b"SecretCameraMaker" in raw           # the upload really contains it...
    out = process_image(raw)
    assert b"SecretCameraMaker" not in out.data  # ...and what we store does not


def test_same_picture_gives_same_hash():
    raw = encode(Image.new("RGB", (60, 60), "purple"))
    assert process_image(raw).sha256 == process_image(raw).sha256


# ------------------------------------------------------------------ alt text
def test_alt_text_rules():
    assert validate_alt_text("  " + ALT + "  ") == ALT
    with pytest.raises(ImageError, match="Describe"):
        validate_alt_text("short")
    with pytest.raises(ImageError, match="too long"):
        validate_alt_text("x" * 501)


# ------------------------------------------------------------------ storing
def test_set_add_then_replace_keeps_one_image_per_role(db, seeded, admin):
    q = make_question(db)
    raw1 = encode(Image.new("RGB", (80, 80), "red"))
    raw2 = encode(Image.new("RGB", (90, 90), "blue"))
    set_question_image(db, q.id, admin, "stem", raw1, ALT, "first.png")
    set_question_image(db, q.id, admin, "stem", raw2, ALT, "second.png")  # replace

    rows = db.scalars(select(QuestionImage)).all()
    assert len(rows) == 1 and rows[0].filename == "second.png" and rows[0].width == 90
    db.refresh(q)
    assert q.version == 3  # created at 1, +1 per change
    notes = [e.note for e in db.scalars(select(QuestionReview).order_by(QuestionReview.id))]
    assert "added" in notes[0] and "replaced" in notes[1]


def test_option_images_and_listing_order(db, seeded, admin):
    q = make_question(db)
    raw = encode(Image.new("RGB", (40, 40), "black"))
    for role in ("C", "stem", "A"):
        set_question_image(db, q.id, admin, role, raw, ALT, f"{role}.png")
    infos = list_image_infos(db, [q.id])[q.id]
    assert [i.role for i in infos] == ["stem", "A", "C"]
    assert infos[0].id is not None and infos[0].data is None  # metadata only, no bytes


def test_get_image_bytes_round_trip(db, seeded, admin):
    q = make_question(db)
    set_question_image(db, q.id, admin, "stem", encode(Image.new("RGB", (40, 40), "red")), ALT, "x.png")
    image_id = db.scalars(select(QuestionImage.id)).one()
    assert get_image_bytes(db, image_id).startswith(b"\x89PNG")


def test_remove_image(db, seeded, admin):
    q = make_question(db)
    set_question_image(db, q.id, admin, "A", encode(Image.new("RGB", (40, 40), "red")), ALT, "x.png")
    remove_question_image(db, db.scalars(select(QuestionImage.id)).one(), admin)
    assert db.scalars(select(QuestionImage)).all() == []
    assert db.scalars(select(QuestionReview).order_by(QuestionReview.id.desc())).first().action == "image_remove"


def test_published_questions_cannot_have_images_changed(db, seeded, admin):
    q = make_question(db, status=S.PUBLISHED)
    with pytest.raises(ImageError, match="draft or pending"):
        set_question_image(db, q.id, admin, "stem", encode(Image.new("RGB", (40, 40))), ALT, "x.png")


def test_student_cannot_change_images(db, seeded, student):
    q = make_question(db)
    with pytest.raises(PermissionDenied):
        set_question_image(db, q.id, student, "stem", encode(Image.new("RGB", (40, 40))), ALT, "x.png")


def test_bad_image_changes_nothing(db, seeded, admin):
    q = make_question(db)
    with pytest.raises(ImageError):
        set_question_image(db, q.id, admin, "stem", b"not an image", ALT, "x.png")
    assert db.scalars(select(QuestionImage)).all() == []
    db.refresh(q)
    assert q.version == 1


def test_unknown_role_rejected(db, seeded, admin):
    q = make_question(db)
    with pytest.raises(ImageError, match="Unknown image position"):
        set_question_image(db, q.id, admin, "Z", encode(Image.new("RGB", (40, 40))), ALT, "x.png")
