from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from typing import Optional

import os
os.makedirs("./data", exist_ok=True)

SQLALCHEMY_DATABASE_URL = "sqlite:///./data/emotions_local.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = "users"

    session_id = Column(String, primary_key=True, index=True)
    trust_score = Column(Integer, default=100)
    last_active = Column(DateTime, default=datetime.datetime.now)
    region = Column(String)
    sub_county = Column(String, nullable=True)
    county = Column(String, nullable=True)
    preferences = Column(Text, default="{}") # JSON string of user preferences (long-term memory)

class FeedbackLog(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    query = Column(Text)
    response = Column(Text)
    score = Column(Integer) # 1 for 👍, -1 for 👎
    correction = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)

class ExpertBrainData(Base):
    __tablename__ = "expert_brain"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, index=True)
    answer = Column(Text)
    source = Column(String)
    embedding_id = Column(Integer, nullable=True) # Reference to FAISS index if needed

# Initialize database
Base.metadata.create_all(bind=engine)

def update_trust_score(session_id: str, delta: int) -> None:
    """Updates the trust score of a user by a given delta."""
    db: Session = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        if user:
            user.trust_score += delta
            if user.trust_score < 0:
                user.trust_score = 0
            user.last_active = datetime.datetime.now()
            db.commit()
    finally:
        db.close()

def create_user_profile(session_id: str, region: str, sub_county: Optional[str] = None, county: Optional[str] = None) -> None:
    """Creates a new user profile in the database."""
    db: Session = SessionLocal()
    try:
        user = UserProfile(
            session_id=session_id, 
            region=region, 
            sub_county=sub_county, 
            county=county,
            trust_score=100,
            last_active=datetime.datetime.now()
        )
        db.add(user)
        db.commit()
    finally:
        db.close()

def get_trust_score(session_id: str) -> int:
    """Returns the trust score for a given session ID."""
    db: Session = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        return int(user.trust_score) if user else 100
    finally:
        db.close()

def log_feedback(session_id: str, query: str, response: str, score: int, correction: Optional[str] = None) -> None:
    db: Session = SessionLocal()
    try:
        log = FeedbackLog(
            session_id=session_id,
            query=query,
            response=response,
            score=score,
            correction=correction
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

def update_preferences(session_id: str, new_prefs: str) -> None:
    db: Session = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        if user:
            user.preferences = new_prefs
            db.commit()
    finally:
        db.close()

def get_preferences(session_id: str) -> str:
    db: Session = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        return user.preferences if user else "{}"
    finally:
        db.close()

