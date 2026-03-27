import os

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DB_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    job_url = Column(Text, nullable=False)
    filename = Column(String(255), nullable=True)
    resume_text = Column(Text, nullable=True)
    status = Column(String(50), nullable=False)  # "success" or "error"
    error_message = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def log_interaction(job_url: str, filename: str, resume_text: str, status: str, error_message: str = None):
    db = SessionLocal()
    try:
        record = Interaction(
            job_url=job_url,
            filename=filename,
            resume_text=resume_text,
            status=status,
            error_message=error_message
        )
        db.add(record)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
