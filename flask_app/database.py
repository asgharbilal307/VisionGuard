# flask_app/database.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(256), nullable=False)
    label = Column(String(50), nullable=False)  # AI-Generated / Real
    confidence = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    threshold = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_size_kb = Column(Float, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)


def init_db(app):
    db_url = os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")

    # SQLite path fix for relative paths
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        db_path = db_url.replace("sqlite:///", "")
        abs_path = os.path.join(app.root_path, "..", db_path)
        db_url = f"sqlite:///{os.path.abspath(abs_path)}"

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session


def log_prediction(
    Session,
    filename,
    label,
    confidence,
    model_name,
    threshold,
    file_size_kb=None,
    width=None,
    height=None,
):
    session = Session()
    try:
        record = Prediction(
            filename=filename,
            label=label,
            confidence=round(confidence, 4),
            model_name=model_name,
            threshold=threshold,
            file_size_kb=file_size_kb,
            image_width=width,
            image_height=height,
        )
        session.add(record)
        session.commit()
        return record.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_history(Session, limit=50):
    session = Session()
    try:
        records = (
            session.query(Prediction)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .all()
        )
        return records
    finally:
        session.close()
