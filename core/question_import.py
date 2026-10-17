"""
core/question_import.py
-----------------------
Turns an uploaded CSV (plus optional diagram files) into validated questions,
with a row-by-row report.

The pipeline (each step is its own function so each can be tested alone):

    bytes --parse_csv--> rows --validate_rows--> ImportReport --save_valid_rows--> database
                                  ^                    |
                       uploaded diagram files          +--> errors_csv (rows to fix and re-upload)

Design goals:
  * ONE BAD ROW MUST NOT REJECT THE FILE. Every row is judged on its own,
    and the admin sees exactly which row failed and why.
  * NOTHING is saved until the admin has seen the preview and clicks Save.
  * Uploaded content is UNTRUSTED. We cap file sizes and counts, validate every
    field with the same Pydantic model used everywhere else, clean every image
    (see core/images.py), and neutralise spreadsheet "formula injection" in
    files we send back.
"""
import csv
import io
import re
from dataclasses import dataclass, field
from difflib import get_close_matches

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.images import (
    ImageError, ProcessedImage, attach_image, clean_filename, process_image, validate_alt_text,
)
from core.orm import Competency, Question, QuestionReview, QuestionStatus, Topic
from core.permissions import ensure_admin
from core.schemas import QuestionIn

MAX_FILE_BYTES = 2 * 1024 * 1024   # 2 MB is far more than 500 questions need
MAX_ROWS = 500
MAX_IMAGE_FILES = 100              # diagram files accepted alongside one CSV

REQUIRED_COLUMNS = [
    "topic", "competency", "question",
    "option_a", "option_b", "option_c", "option_d",
    "correct_option", "difficulty",
]
OPTIONAL_COLUMNS = ["explanation", "source", "image_file", "image_alt"]
TEMPLATE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Be forgiving about common header spellings (admins will use their own words).
HEADER_ALIASES = {
    "question_text": "question", "text": "question", "stem": "question",
    "correct_answer": "correct_option", "answer": "correct_option", "correct": "correct_option",
    "level": "difficulty",
    "image": "image_file", "diagram": "image_file", "image_description": "image_alt",
}

# Friendly names for error messages (Pydantic field name -> CSV column name).
FIELD_LABELS = {"text": "question", "options": "options (option_a to option_d)"}


class ImportFileError(Exception):
    """The WHOLE FILE is unusable (empty, too big, wrong columns). Row problems
    are not exceptions: they are recorded per row in the report."""


# ============================================================ report objects
@dataclass
class RowResult:
    row_number: int                      # matches the spreadsheet row (header = row 1)
    raw: dict                            # what the admin typed, for the "fix and re-upload" file
    question: QuestionIn | None = None   # set only when the row validated
    competency_id: int | None = None
    image: ProcessedImage | None = None  # the cleaned diagram, if the row has one
    image_filename: str = ""
    image_alt: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ImportReport:
    results: list[RowResult]
    unused_images: list[str] = field(default_factory=list)  # uploaded but no row refers to them

    @property
    def ok_rows(self) -> list[RowResult]:
        return [r for r in self.results if r.ok]

    @property
    def error_rows(self) -> list[RowResult]:
        return [r for r in self.results if not r.ok]


@dataclass
class ParsedCsv:
    rows: list[tuple[int, dict]]         # (row_number, {column: value})
    ignored_columns: list[str]           # columns we did not recognise (maybe a typo!)


# ============================================================ step 1: parse
def _detect_delimiter(header_line: str) -> str:
    """Excel in some regions saves CSV with ';' or tabs instead of commas."""
    counts = {d: header_line.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _normalise_header(name: str) -> str:
    key = re.sub(r"[\s\-]+", "_", (name or "").strip().lower())
    return HEADER_ALIASES.get(key, key)


def parse_csv(data: bytes) -> ParsedCsv:
    """Read raw bytes into rows. Raises ImportFileError if the file is unusable."""
    if not data or not data.strip():
        raise ImportFileError("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ImportFileError("The file is larger than 2 MB. Split it into smaller files.")
    try:
        # utf-8-sig silently removes the invisible BOM that Excel adds.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ImportFileError(
            "The file is not UTF-8. In Excel choose Save As > 'CSV UTF-8 (Comma delimited)'."
        )

    delimiter = _detect_delimiter(text.splitlines()[0])
    rows: list[tuple[int, dict]] = []
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        headers = [_normalise_header(h) for h in next(reader)]
        if len(set(headers)) != len(headers):
            raise ImportFileError("Two columns have the same name. Each column must be unique.")
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            raise ImportFileError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Columns found: {', '.join(h for h in headers if h) or 'none'}. "
                "Download the template to see the expected columns."
            )
        # enumerate from 2 because row 1 is the header: numbers match the spreadsheet.
        for row_number, cells in enumerate(reader, start=2):
            if not any(c.strip() for c in cells):
                continue  # skip completely blank lines
            record = {h: (cells[i].strip() if i < len(cells) else "") for i, h in enumerate(headers)}
            rows.append((row_number, record))
            if len(rows) > MAX_ROWS:
                raise ImportFileError(f"Too many rows (limit is {MAX_ROWS} per file).")
    except csv.Error as e:
        raise ImportFileError(f"The file could not be read as CSV: {e}")

    if not rows:
        raise ImportFileError("The file has a header row but no questions.")
    ignored = [h for h in headers if h and h not in TEMPLATE_COLUMNS]
    return ParsedCsv(rows=rows, ignored_columns=ignored)


# ============================================================ step 2: validate
def normalise_text(text: str) -> str:
    """Canonical form used to spot duplicates: case, spacing and a trailing
    '?' or '.' don't make a question 'different'."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip("?. ")


def format_validation_error(e: ValidationError) -> list[str]:
    """Turn Pydantic's technical errors into messages an admin understands."""
    messages = []
    for err in e.errors():
        msg = err["msg"].removeprefix("Value error, ")
        if err["loc"]:  # a problem with one field
            name = err["loc"][0]
            messages.append(f"{FIELD_LABELS.get(name, name)}: {msg}")
        else:           # a rule involving several fields (e.g. correct answer not in options)
            messages.append(msg)
    return messages


def _build_payload(raw: dict) -> tuple[dict, list[str]]:
    """Map CSV columns onto the QuestionIn fields."""
    problems: list[str] = []
    try:
        as_float = float(raw.get("difficulty", ""))
        if as_float != int(as_float):
            raise ValueError
        difficulty = int(as_float)
    except (ValueError, OverflowError):
        problems.append("difficulty: must be a whole number from 1 to 5")
        difficulty = 3  # placeholder so Pydantic doesn't report the same problem twice

    payload = {
        "topic": raw.get("topic", ""),
        "competency": raw.get("competency", ""),
        "text": raw.get("question", ""),
        "options": {k.upper(): raw.get(f"option_{k}", "") for k in "abcd"},
        "correct_option": raw.get("correct_option", ""),
        "difficulty": difficulty,
        "explanation": raw.get("explanation", ""),
        "source": (raw.get("source") or "original").strip().lower().replace(" ", "_").replace("-", "_"),
    }
    return payload, problems


def validate_rows(db: Session, parsed: ParsedCsv, images: dict[str, bytes] | None = None) -> ImportReport:
    """Check every row. Reads the database (topics, existing questions) but never writes.

    `images` maps the uploaded diagram file NAME to its bytes. A row points at
    its diagram through the `image_file` column.
    """
    images = images or {}
    if len(images) > MAX_IMAGE_FILES:
        raise ImportFileError(f"Too many image files (limit is {MAX_IMAGE_FILES} per upload).")
    # Match file names ignoring capitals and any folder prefix ("Diagrams\Circuit1.PNG" -> "circuit1.png").
    uploaded = {clean_filename(name).lower(): (name, data) for name, data in images.items()}
    processed_cache: dict[str, ProcessedImage | ImageError] = {}  # clean each file only once
    used_images: set[str] = set()

    topics = {t.name.lower(): t.name for t in db.scalars(select(Topic))}
    competencies: dict[tuple[str, str], int] = {}          # (topic, competency) -> id
    comps_by_topic: dict[str, list[str]] = {}
    for topic_name, comp_name, comp_id in db.execute(
        select(Topic.name, Competency.name, Competency.id).join(Competency, Competency.topic_id == Topic.id)
    ):
        competencies[(topic_name.lower(), comp_name.lower())] = comp_id
        comps_by_topic.setdefault(topic_name.lower(), []).append(comp_name)

    existing = {normalise_text(t): qid for qid, t in db.execute(select(Question.id, Question.text))}
    seen_in_file: dict[str, int] = {}

    results: list[RowResult] = []
    for row_number, raw in parsed.rows:
        res = RowResult(row_number=row_number, raw=raw)
        payload, res.errors = _build_payload(raw)

        # --- field-level rules (the same QuestionIn model used everywhere) ---
        try:
            res.question = QuestionIn(**payload)
        except ValidationError as e:
            res.errors.extend(format_validation_error(e))

        # --- does the topic / competency exist? (with a "did you mean" hint) ---
        topic_in, comp_in = payload["topic"].strip(), payload["competency"].strip()
        if topic_in:
            canonical = topics.get(topic_in.lower())
            if canonical is None:
                hint = get_close_matches(topic_in, list(topics.values()), n=1, cutoff=0.6)
                res.errors.append(
                    f"topic: '{topic_in}' is not in the syllabus"
                    + (f" (did you mean '{hint[0]}'?)" if hint else "")
                )
            elif comp_in:
                res.competency_id = competencies.get((topic_in.lower(), comp_in.lower()))
                if res.competency_id is None:
                    hint = get_close_matches(comp_in, comps_by_topic.get(topic_in.lower(), []), n=1, cutoff=0.5)
                    res.errors.append(
                        f"competency: '{comp_in}' is not under topic '{canonical}'"
                        + (f" (did you mean '{hint[0]}'?)" if hint else "")
                    )

        # --- the diagram: was the file uploaded, is it a real image, is it described? ---
        image_name, image_alt = raw.get("image_file", "").strip(), raw.get("image_alt", "").strip()
        if image_name:
            key = clean_filename(image_name).lower()
            if key not in uploaded:
                res.errors.append(f"image_file: '{image_name}' was not uploaded with the CSV")
            else:
                used_images.add(key)
                if key not in processed_cache:
                    try:
                        processed_cache[key] = process_image(uploaded[key][1])
                    except ImageError as e:
                        processed_cache[key] = e
                outcome = processed_cache[key]
                if isinstance(outcome, ImageError):
                    res.errors.append(f"image_file: '{image_name}': {outcome}")
                else:
                    res.image, res.image_filename = outcome, clean_filename(image_name)
            try:
                res.image_alt = validate_alt_text(image_alt)
            except ImageError as e:
                res.errors.append(f"image_alt: {e}")
        elif image_alt:
            res.warnings.append("image_alt is filled in but image_file is empty, so no diagram will be attached.")

        # --- duplicates: against the database, then against earlier rows in this file ---
        key = normalise_text(payload["text"])
        if key:
            if key in existing:
                res.errors.append(f"question: duplicate of existing question #{existing[key]}")
            elif key in seen_in_file:
                res.errors.append(f"question: duplicate of row {seen_in_file[key]} in this file")

        # --- warnings: allowed through, but worth a human's attention ---
        if res.ok and not res.question.explanation:
            res.warnings.append("No explanation. Add one before this can be published.")

        if res.ok and key:
            seen_in_file[key] = row_number  # only register rows that are themselves valid
        results.append(res)

    unused = sorted(name for key, (name, _) in uploaded.items() if key not in used_images)
    return ImportReport(results, unused_images=unused)


# ============================================================ step 3: save
def save_valid_rows(db: Session, report: ImportReport, actor: CurrentUser, filename: str) -> int:
    """Save only the valid rows, as PENDING_REVIEW. All-or-nothing for those rows.

    Imported questions NEVER go straight to published: a human must approve
    them in the review queue first. Each one gets an audit-log entry saying
    which file and row it came from.
    """
    ensure_admin(actor)  # defence in depth: the core re-checks the role
    safe_name = filename.strip()[:100]
    saved = 0
    try:
        for r in report.ok_rows:
            q = r.question
            question = Question(
                competency_id=r.competency_id, text=q.text, options=q.options,
                correct_option=q.correct_option, difficulty=q.difficulty,
                explanation=q.explanation, source=q.source,
                status=QuestionStatus.PENDING_REVIEW, created_by=actor.id,
            )
            db.add(question)
            db.flush()  # assigns question.id so images and the log entry can point to it
            if r.image is not None:
                attach_image(db, question.id, "stem", r.image, r.image_alt, r.image_filename)
            db.add(QuestionReview(
                question_id=question.id, actor_id=actor.id, action="import",
                from_status="-", to_status=QuestionStatus.PENDING_REVIEW.value,
                note=f"Imported from {safe_name}, row {r.row_number}",
            ))
            saved += 1
        db.commit()
    except Exception:
        db.rollback()  # never leave half a file saved
        raise
    return saved


# ============================================================ files we hand back
def _safe_cell(value):
    """CSV/formula injection defence. If a cell starts with = + - @ then Excel
    may run it as a formula when the admin opens the file. A leading apostrophe
    makes Excel treat it as plain text."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _to_csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_safe_cell(c) for c in row])
    # The BOM tells Excel the file is UTF-8, so symbols like Ω, µ, ² and ° survive.
    return "\ufeff" + buf.getvalue()


def template_csv() -> str:
    return _to_csv(TEMPLATE_COLUMNS, [
        ["Current Electricity", "Apply Ohm's law",
         "A 6 V battery is connected across a 2 Ω resistor. What current flows through the resistor?",
         "0.33 A", "3 A", "12 A", "8 A", "B", 1,
         "Use V = IR, so I = V/R = 6/2 = 3 A.", "original", "", ""],
        ["Motion", "Apply equations of uniform acceleration",
         "A body moving at 10 m/s decelerates uniformly to rest in 4 s. What is the magnitude of its deceleration?",
         "0.4 m/s²", "2.5 m/s²", "6 m/s²", "40 m/s²", "B", 2,
         "Use a = (v - u)/t = (0 - 10)/4 = -2.5 m/s², so the magnitude is 2.5 m/s².", "original", "", ""],
    ])


def errors_csv(report: ImportReport) -> str:
    """The rows that failed, plus an 'errors' column, ready to fix and re-upload."""
    rows = [[r.raw.get(c, "") for c in TEMPLATE_COLUMNS] + ["; ".join(r.errors)] for r in report.error_rows]
    return _to_csv(TEMPLATE_COLUMNS + ["errors"], rows)
