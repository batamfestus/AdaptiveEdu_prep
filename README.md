# AdaptPrep 🎯

An adaptive exam-preparation platform for **JAMB UTME Physics**, built as a learning project.
It follows a continuous learning loop:

`Diagnostic -> Readiness Score -> Personalised Plan -> Adaptive Practice -> AI Explanation -> Competency Update -> Reassessment`

## Status
- [x] Milestone 1a: project scaffold, database schema, syllabus seed, question validator
- [ ] Milestone 1b: authentication + roles, admin question upload, review queue
- [ ] Milestone 2: CBT test flow, grading, AI explanations
- [ ] Milestone 3: dashboard, adaptive scheduler, chat history

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional: add keys later
python -m scripts.init_db          # creates tables + loads the syllabus
streamlit run app.py
pytest -q                          # run the tests
```

## Architecture (so far)
```
app.py / pages/   -> Streamlit UI only (thin)
core/             -> all logic; never imports Streamlit
  config.py       -> settings
  db.py           -> engine + sessions
  orm.py          -> database tables
  schemas.py      -> Pydantic validation
data/             -> syllabus seed
scripts/          -> one-off setup scripts
tests/            -> pytest tests
```

## Design decisions
- `core/` is UI-free so the logic is testable and reusable.
- Questions pass through a review status (`draft -> pending_review -> approved -> published`); AI-drafted content is never shown to students unreviewed.
