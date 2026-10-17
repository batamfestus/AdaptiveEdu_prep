"""
core/images.py
--------------
Everything about question diagrams: validating and cleaning uploaded images,
storing them, replacing and removing them.

IMPORTANT: an uploaded image is UNTRUSTED INPUT, just like a CSV. We never store
the file we were given. We DECODE it with Pillow and RE-ENCODE a brand-new image:

  * We check the real format by decoding it, not the filename or the browser's
    claimed content-type (both can be faked).
  * SVG is refused: it is XML and can carry scripts.
  * Re-encoding strips hidden metadata (camera info, GPS location) and anything
    else smuggled inside the file.
  * Huge images (a "decompression bomb": tiny file, billions of pixels) are
    refused before any pixels are decoded, so they can't exhaust our memory.
  * Transparent backgrounds are flattened onto WHITE. Otherwise a black line
    drawing on a transparent PNG becomes invisible in dark mode.
"""
import hashlib
import io
import os
from dataclasses import dataclass

from PIL import Image, ImageOps

from core.auth import CurrentUser
from core.orm import EDITABLE_STATUSES, Question, QuestionImage, QuestionReview
from core.permissions import ensure_admin
from sqlalchemy import select
from sqlalchemy.orm import Session

MAX_UPLOAD_BYTES = 5 * 1024 * 1024     # reject files bigger than this before even opening them
MAX_PIXELS = 25_000_000                # width x height limit (5000 x 5000)
MAX_SIDE = 1200                        # stored images are shrunk to fit inside 1200 x 1200
JPEG_THRESHOLD = 600_000               # photos whose PNG is bigger than this are stored as JPEG
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
VALID_ROLES = ("stem", "A", "B", "C", "D")  # stem = the diagram in the question itself
MIN_ALT_LENGTH = 10
MAX_ALT_LENGTH = 500


class ImageError(Exception):
    """A problem with an image. The message is safe to show the admin."""


@dataclass(frozen=True)
class ProcessedImage:
    """A clean, ready-to-store image (all fields are produced by us, not the uploader)."""
    data: bytes
    content_type: str
    width: int
    height: int
    sha256: str


@dataclass(frozen=True)
class ImageInfo:
    """What the UI needs to SHOW an image. `data` is filled only for previews of
    images that are not saved yet; for saved ones the UI loads bytes by id (cached)."""
    id: int | None
    role: str
    alt_text: str
    width: int
    height: int
    sha256: str
    data: bytes | None = None


# ============================================================ cleaning an upload
def _flatten_on_white(img: Image.Image) -> Image.Image:
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def process_image(raw: bytes) -> ProcessedImage:
    """Validate an uploaded image and return a cleaned copy, or raise ImageError."""
    if not raw:
        raise ImageError("The image file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageError("The image is larger than 5 MB. Please shrink it first.")

    # Step 1: read ONLY the header (format and size). No pixels are decoded yet.
    try:
        probe = Image.open(io.BytesIO(raw))
        fmt, (width, height) = probe.format, probe.size
    except Exception:  # Pillow raises many different errors for bad files
        raise ImageError("This file is not a valid image. Use PNG, JPEG or WebP.")
    if fmt not in ALLOWED_FORMATS:
        raise ImageError(f"{fmt or 'This'} images are not allowed. Use PNG, JPEG or WebP.")
    if width * height > MAX_PIXELS:
        raise ImageError(f"The image is too large ({width} x {height} pixels). Maximum is 5000 x 5000.")

    # Step 2: now it is safe to decode fully. This also catches truncated/corrupt files.
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise ImageError("The image file is damaged and could not be read.")

    img = ImageOps.exif_transpose(img)            # honour phone-camera rotation, then metadata is dropped
    img = _flatten_on_white(img)
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)  # shrinks only, never enlarges

    buffer = io.BytesIO()
    img.save(buffer, "PNG", optimize=True)        # PNG keeps line drawings crisp
    data, content_type = buffer.getvalue(), "image/png"
    if len(data) > JPEG_THRESHOLD:                # photographs are far smaller as JPEG
        jpeg = io.BytesIO()
        img.save(jpeg, "JPEG", quality=85, optimize=True)
        if len(jpeg.getvalue()) < len(data):
            data, content_type = jpeg.getvalue(), "image/jpeg"

    return ProcessedImage(data, content_type, img.width, img.height, hashlib.sha256(data).hexdigest())


def validate_alt_text(alt_text: str) -> str:
    alt = (alt_text or "").strip()
    if len(alt) < MIN_ALT_LENGTH:
        raise ImageError(
            f"Describe the image in words (at least {MIN_ALT_LENGTH} characters). "
            "This helps screen readers, reviewers and the AI tutor."
        )
    if len(alt) > MAX_ALT_LENGTH:
        raise ImageError(f"The image description is too long (maximum {MAX_ALT_LENGTH} characters).")
    return alt


def clean_filename(name: str) -> str:
    """Keep only the file's own name (no folders), at most 120 characters."""
    return os.path.basename((name or "").replace("\\", "/")).strip()[:120]


# ============================================================ storing
def attach_image(db: Session, question_id: int, role: str, image: ProcessedImage,
                 alt_text: str, filename: str) -> QuestionImage:
    """Add an already-cleaned image to a question. Does NOT commit: the caller
    decides the transaction, so a question and its images are saved together or not at all."""
    row = QuestionImage(
        question_id=question_id, role=role, filename=clean_filename(filename),
        content_type=image.content_type, width=image.width, height=image.height,
        sha256=image.sha256, alt_text=alt_text, data=image.data,
    )
    db.add(row)
    return row


def set_question_image(db: Session, question_id: int, actor: CurrentUser, role: str,
                       raw: bytes, alt_text: str, filename: str) -> None:
    """Add OR replace the image for one role on an existing question."""
    ensure_admin(actor)
    if role not in VALID_ROLES:
        raise ImageError(f"Unknown image position '{role}'.")
    question = db.get(Question, question_id)
    if question is None:
        raise ImageError("That question no longer exists.")
    if question.status not in EDITABLE_STATUSES:
        raise ImageError("Images can only be changed on draft or pending questions. Send the question back first.")

    processed = process_image(raw)       # validate BEFORE touching the database
    alt = validate_alt_text(alt_text)

    existing = db.scalar(select(QuestionImage).where(
        QuestionImage.question_id == question_id, QuestionImage.role == role))
    if existing is not None:
        db.delete(existing)
        db.flush()  # the delete must reach the database BEFORE the insert, or the UNIQUE rule complains
    attach_image(db, question_id, role, processed, alt, filename)

    question.version += 1
    db.add(QuestionReview(
        question_id=question_id, actor_id=actor.id, action="image_set",
        from_status=question.status.value, to_status=question.status.value,
        note=f"{role} image {'replaced' if existing else 'added'}: {clean_filename(filename)}",
    ))
    db.commit()


def remove_question_image(db: Session, image_id: int, actor: CurrentUser) -> None:
    ensure_admin(actor)
    image = db.get(QuestionImage, image_id)
    if image is None:
        raise ImageError("That image no longer exists.")
    question = db.get(Question, image.question_id)
    if question.status not in EDITABLE_STATUSES:
        raise ImageError("Images can only be changed on draft or pending questions. Send the question back first.")

    role = image.role
    db.delete(image)
    question.version += 1
    db.add(QuestionReview(
        question_id=question.id, actor_id=actor.id, action="image_remove",
        from_status=question.status.value, to_status=question.status.value,
        note=f"{role} image removed",
    ))
    db.commit()


# ============================================================ reading
def get_image_bytes(db: Session, image_id: int) -> bytes:
    value = db.scalar(select(QuestionImage.data).where(QuestionImage.id == image_id))
    if value is None:
        raise ImageError("That image no longer exists.")
    return bytes(value)  # Postgres hands back a memoryview; the UI wants plain bytes


def list_image_infos(db: Session, question_ids: list[int]) -> dict[int, list[ImageInfo]]:
    """Metadata only (no bytes), grouped by question. Stem image first, then A to D."""
    if not question_ids:
        return {}
    stmt = select(
        QuestionImage.id, QuestionImage.question_id, QuestionImage.role, QuestionImage.alt_text,
        QuestionImage.width, QuestionImage.height, QuestionImage.sha256,
    ).where(QuestionImage.question_id.in_(question_ids))
    grouped: dict[int, list[ImageInfo]] = {}
    for image_id, qid, role, alt, w, h, sha in db.execute(stmt):
        grouped.setdefault(qid, []).append(ImageInfo(image_id, role, alt, w, h, sha))
    for infos in grouped.values():
        infos.sort(key=lambda i: VALID_ROLES.index(i.role))
    return grouped
