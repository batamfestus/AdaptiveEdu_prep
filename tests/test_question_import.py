"""
Tests for core/question_import.py. Run with:  pytest -q

Pattern used throughout: start from ONE valid row (good_row), break exactly
ONE thing per test, and assert on the error message the admin would see.
"""
import csv
import io

import pytest
from PIL import Image
from sqlalchemy import select

from core import question_import as qi
from core.orm import Question, QuestionImage, QuestionReview, QuestionStatus
from core.permissions import PermissionDenied
from core.question_import import (
    ImportFileError, errors_csv, parse_csv, save_valid_rows, template_csv, validate_rows,
)

HEADER = qi.TEMPLATE_COLUMNS


def good_row(**overrides) -> list:
    row = {
        "topic": "Current Electricity", "competency": "Apply Ohm's law",
        "question": "A 12 V battery drives a current through a 4 ohm resistor. Find the current.",
        "option_a": "48 A", "option_b": "3 A", "option_c": "8 A", "option_d": "0.33 A",
        "correct_option": "B", "difficulty": "2",
        "explanation": "I = V/R = 12/4 = 3 A.", "source": "original",
        "image_file": "", "image_alt": "",
    }
    row.update(overrides)
    return [row[c] for c in HEADER]


def make_csv(*rows, header=HEADER, delimiter=",", bom=False) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter)
    writer.writerow(header)
    writer.writerows(rows)
    return (("\ufeff" if bom else "") + buf.getvalue()).encode("utf-8")


def run(db, *rows, images=None, **kwargs):
    return validate_rows(db, parse_csv(make_csv(*rows, **kwargs)), images)


def png_bytes(size=(300, 200), color=(30, 90, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ parsing
def test_template_itself_is_valid(db, seeded):
    report = validate_rows(db, parse_csv(template_csv().encode("utf-8")))
    assert len(report.ok_rows) == 2 and not report.error_rows


def test_bom_and_semicolon_delimiter_are_handled(db, seeded):
    report = run(db, good_row(), delimiter=";", bom=True)
    assert len(report.ok_rows) == 1


def test_missing_required_column_is_a_file_error():
    header = [c for c in HEADER if c != "correct_option"]
    with pytest.raises(ImportFileError, match="correct_option"):
        parse_csv(make_csv(header=header))


def test_header_aliases_work(db, seeded):
    header = ["Topic", "Competency", "Question Text", "Option A", "Option B", "Option C",
              "Option D", "Correct Answer", "Level", "Explanation", "Source"]
    assert len(run(db, good_row(), header=header).ok_rows) == 1


def test_empty_file_and_header_only_file():
    with pytest.raises(ImportFileError, match="empty"):
        parse_csv(b"   ")
    with pytest.raises(ImportFileError, match="no questions"):
        parse_csv(make_csv())


def test_non_utf8_file_rejected():
    with pytest.raises(ImportFileError, match="UTF-8"):
        parse_csv("topic,competency\n\xe9".encode("latin-1"))


def test_size_and_row_limits(monkeypatch):
    monkeypatch.setattr(qi, "MAX_ROWS", 2)
    with pytest.raises(ImportFileError, match="Too many rows"):
        parse_csv(make_csv(good_row(), good_row(), good_row()))
    monkeypatch.setattr(qi, "MAX_FILE_BYTES", 50)
    with pytest.raises(ImportFileError, match="larger"):
        parse_csv(make_csv(good_row()))


def test_unrecognised_columns_are_reported():
    parsed = parse_csv(make_csv(good_row() + ["x"], header=HEADER + ["explaination"]))
    assert parsed.ignored_columns == ["explaination"]


# ------------------------------------------------------------------ validation
def test_valid_row_passes(db, seeded):
    report = run(db, good_row())
    assert report.ok_rows[0].question.correct_option == "B"
    assert report.ok_rows[0].competency_id is not None


def test_topic_typo_gets_a_suggestion(db, seeded):
    report = run(db, good_row(topic="Current Electrcity"))
    assert "did you mean 'Current Electricity'" in report.error_rows[0].errors[0]


def test_competency_must_belong_to_the_topic(db, seeded):
    report = run(db, good_row(competency="Apply equations of uniform acceleration"))
    assert "not under topic" in report.error_rows[0].errors[0]


def test_row_numbers_match_the_spreadsheet(db, seeded):
    report = run(db, good_row(), good_row(correct_option="E", question="A different question about circuits?"))
    assert [r.row_number for r in report.results] == [2, 3]  # header is row 1
    assert not report.results[1].ok


def test_bad_difficulty_gives_one_clear_error(db, seeded):
    errors = run(db, good_row(difficulty="hard")).error_rows[0].errors
    assert len(errors) == 1 and errors[0].startswith("difficulty")


def test_one_bad_row_does_not_block_the_others(db, seeded):
    report = run(db, good_row(), good_row(correct_option="Z", question="Another valid sounding question here"))
    assert len(report.ok_rows) == 1 and len(report.error_rows) == 1


def test_duplicate_inside_the_file(db, seeded):
    report = run(db, good_row(), good_row(question="  a 12 v BATTERY drives a current through a 4 ohm resistor. find the current  "))
    assert "duplicate of row 2" in report.error_rows[0].errors[0]


def test_duplicate_of_a_saved_question(db, seeded, admin):
    save_valid_rows(db, run(db, good_row()), admin, "first.csv")
    report = run(db, good_row())
    assert "duplicate of existing question" in report.error_rows[0].errors[0]


def test_missing_explanation_is_only_a_warning(db, seeded):
    report = run(db, good_row(explanation=""))
    assert report.ok_rows and report.ok_rows[0].warnings


def test_bad_source_rejected(db, seeded):
    assert run(db, good_row(source="copied from a book")).error_rows


# ------------------------------------------------------------------ saving
def test_save_creates_pending_questions_and_audit_log(db, seeded, admin):
    report = run(db, good_row(), good_row(correct_option="Z", question="Another valid sounding question here"))
    assert save_valid_rows(db, report, admin, "my file.csv") == 1

    saved = db.scalars(select(Question)).all()
    assert len(saved) == 1  # the invalid row was NOT saved
    assert saved[0].status == QuestionStatus.PENDING_REVIEW  # never straight to published
    assert saved[0].created_by == admin.id
    log = db.scalars(select(QuestionReview)).one()
    assert log.action == "import" and "my file.csv, row 2" in log.note


def test_student_cannot_save(db, seeded, student):
    with pytest.raises(PermissionDenied):
        save_valid_rows(db, run(db, good_row()), student, "x.csv")
    assert db.scalars(select(Question)).all() == []


# ------------------------------------------------------------------ files handed back
def test_errors_csv_neutralises_formula_injection(db, seeded):
    evil = '=HYPERLINK("http://evil.example","click")'
    report = run(db, good_row(correct_option="Z", explanation=evil))
    out = errors_csv(report)
    assert out.startswith("\ufeff")  # BOM so Excel keeps symbols like Ω
    cells = list(csv.reader(io.StringIO(out.lstrip("\ufeff"))))[1]
    assert evil not in cells and "'" + evil in cells
    assert cells[-1].startswith("correct_option")  # the errors column


# ------------------------------------------------------------------ diagrams
ALT = "Circuit diagram: a 12 V battery across a 6 ohm and a 3 ohm resistor in parallel."


def test_row_with_uploaded_diagram_is_valid(db, seeded):
    report = run(db, good_row(image_file="circuit.png", image_alt=ALT), images={"circuit.png": png_bytes()})
    row = report.ok_rows[0]
    assert row.image is not None and row.image.content_type == "image/png" and row.image_alt == ALT
    assert report.unused_images == []


def test_filename_matching_ignores_case_and_folders(db, seeded):
    report = run(db, good_row(image_file="Diagrams\\CIRCUIT.PNG", image_alt=ALT), images={"circuit.png": png_bytes()})
    assert report.ok_rows


def test_referenced_image_that_was_not_uploaded(db, seeded):
    report = run(db, good_row(image_file="circuit.png", image_alt=ALT))
    assert "was not uploaded" in report.error_rows[0].errors[0]


def test_diagram_needs_a_description(db, seeded):
    report = run(db, good_row(image_file="circuit.png", image_alt="short"), images={"circuit.png": png_bytes()})
    assert any(e.startswith("image_alt") for e in report.error_rows[0].errors)


def test_fake_image_is_rejected(db, seeded):
    report = run(db, good_row(image_file="evil.png", image_alt=ALT), images={"evil.png": b"<svg onload=alert(1)>"})
    assert "not a valid image" in report.error_rows[0].errors[0]


def test_unused_uploads_are_reported(db, seeded):
    report = run(db, good_row(), images={"spare.png": png_bytes()})
    assert report.unused_images == ["spare.png"]


def test_same_diagram_can_serve_two_rows(db, seeded):
    rows = [good_row(image_file="c.png", image_alt=ALT),
            good_row(question="Another question that uses the very same circuit diagram?", image_file="c.png", image_alt=ALT)]
    assert len(run(db, *rows, images={"c.png": png_bytes()}).ok_rows) == 2


def test_alt_text_without_a_file_is_only_a_warning(db, seeded):
    report = run(db, good_row(image_alt=ALT))
    assert report.ok_rows and report.ok_rows[0].warnings


def test_too_many_image_files(db, seeded, monkeypatch):
    monkeypatch.setattr(qi, "MAX_IMAGE_FILES", 1)
    with pytest.raises(ImportFileError, match="Too many image files"):
        run(db, good_row(), images={"a.png": png_bytes(), "b.png": png_bytes()})


def test_save_stores_the_diagram_with_the_question(db, seeded, admin):
    report = run(db, good_row(image_file="c.png", image_alt=ALT), images={"c.png": png_bytes()})
    save_valid_rows(db, report, admin, "with-image.csv")
    image = db.scalars(select(QuestionImage)).one()
    assert image.role == "stem" and image.alt_text == ALT and image.filename == "c.png"
    assert bytes(image.data).startswith(b"\x89PNG")
