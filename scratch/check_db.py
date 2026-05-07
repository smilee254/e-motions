from api._database import SessionLocal, ExpertBrainData
db = SessionLocal()
count = db.query(ExpertBrainData).count()
print(f"Expert brain count: {count}")
db.close()
