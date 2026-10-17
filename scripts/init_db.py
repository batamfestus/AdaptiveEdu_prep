"""
scripts/init_db.py
------------------
One-off setup script. Run it from the project root:

    python -m scripts.init_db

It (1) creates all tables and (2) loads topics/competencies from
data/syllabus_seed.json. It is IDEMPOTENT: running it twice does not
create duplicates, so it is safe to re-run after editing the seed file.
"""
import json

from core.config import BASE_DIR
from core.db import get_engine, get_session
from core.orm import Base, Competency, Topic

SEED_FILE = BASE_DIR / "data" / "syllabus_seed.json"


def create_tables() -> None:
    # create_all only creates tables that don't exist yet; it never deletes.
    # (For real schema changes later we'd use Alembic migrations.)
    Base.metadata.create_all(get_engine())


def seed_syllabus() -> tuple[int, int]:
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    new_topics = new_competencies = 0

    with get_session() as db:
        for t in data["topics"]:
            topic = db.query(Topic).filter_by(name=t["name"]).one_or_none()
            if topic is None:
                topic = Topic(name=t["name"], exam_weight=t.get("exam_weight", 1.0))
                db.add(topic)
                db.flush()  # flush assigns topic.id without ending the transaction
                new_topics += 1

            existing = {c.name for c in topic.competencies}
            for c in t["competencies"]:
                if c["name"] not in existing:
                    db.add(Competency(topic_id=topic.id, name=c["name"],
                                      description=c.get("description", "")))
                    new_competencies += 1
        db.commit()  # one commit = all-or-nothing

    return new_topics, new_competencies


if __name__ == "__main__":
    create_tables()
    topics, comps = seed_syllabus()
    print(f"Database ready. Added {topics} topics and {comps} competencies.")
