from api._database import SessionLocal, FeedbackLog
import pandas as pd
import os

def view_feedback():
    if not os.path.exists("./data/emotions_local.db"):
        print("Database not found. Make sure the app has run at least once.")
        return

    db = SessionLocal()
    try:
        feedback = db.query(FeedbackLog).all()
        if not feedback:
            print("No feedback collected yet. Chat with Sentinel and use the 👍/👎 buttons!")
        else:
            data = []
            for f in feedback:
                data.append({
                    "Timestamp": f.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Query": (f.query[:50] + '..') if len(f.query) > 50 else f.query,
                    "Score": "👍" if f.score == 1 else "👎",
                    "Correction": f.correction or "N/A"
                })
            
            df = pd.DataFrame(data)
            print("\n" + "="*60)
            print(" SENTINEL FEEDBACK & CORRECTION LOG ".center(60, " "))
            print("="*60)
            print(df.to_string(index=False))
            print("="*60 + "\n")
    finally:
        db.close()

if __name__ == "__main__":
    view_feedback()
