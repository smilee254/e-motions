from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

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

    def __init__(self, session_id: str, region: str, sub_county: str = None, county: str = None):
        self.session_id = session_id
        self.region = region
        self.sub_county = sub_county
        self.county = county
        self.trust_score = 100
        self.last_active = datetime.datetime.now()

# Initialize database
Base.metadata.create_all(bind=engine)

def update_trust_score(session_id: str, delta: int):
    db = SessionLocal()
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

def create_user_profile(session_id: str, region: str, sub_county: str = None, county: str = None):
    db = SessionLocal()
    try:
        user = UserProfile(session_id=session_id, region=region, sub_county=sub_county, county=county)
        db.add(user)
        db.commit()
    finally:
        db.close()

def get_trust_score(session_id: str) -> int:
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        score = user.trust_score if user else 100
        return int(score)
    finally:
        db.close()

