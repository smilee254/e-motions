"""
Sentinel Expert Brain — Full Rebuild Script
============================================
Run this to build the FAISS index + SQLite DB from scratch.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) for embedding.
- No API rate limits
- Fully offline after model download
- Consistent with the runtime embedder in _fallback.py

Usage:
    PYTHONPATH=. python3 api/_build_brain.py
"""

import pandas as pd
from datasets import load_dataset
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from api._database import SessionLocal, ExpertBrainData

# 1. Load Embedder
print("Loading sentence-transformer embedder (all-MiniLM-L6-v2)...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')
DIM = st_model.get_embedding_dimension()
print(f"Embedder ready. Dimension: {DIM}")

# 2. Pull Expert Archives (CounselChat + MentalChat16K)
print("\nDownloading CounselChat & MentalChat archives...")
try:
    ds_counsel = load_dataset("nbertagnolli/counsel-chat")
    ds_mental = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
except Exception as e:
    print(f"Error downloading datasets: {e}")
    exit(1)

# 3. Extract Expert Q&A Pairs
df_counsel = pd.DataFrame(ds_counsel['train'])[['questionText', 'answerText']]

def parse_mental_health(text):
    if "<HUMAN>:" in text and "<ASSISTANT>:" in text:
        parts = text.split("<ASSISTANT>:")
        q = parts[0].replace("<HUMAN>:", "").strip()
        a = parts[1].strip()
        return q, a
    return None, None

mental_rows = [parse_mental_health(row['text']) for row in ds_mental['train']]
df_mental = pd.DataFrame([r for r in mental_rows if r[0]], columns=['questionText', 'answerText'])

expert_archive = pd.concat([df_counsel, df_mental], ignore_index=True).dropna()
print(f"Total expert cases: {len(expert_archive)}")

# 4. Generate Embeddings in Batches
print(f"\nEmbedding {len(expert_archive)} expert questions...")
BATCH_SIZE = 64

all_embeddings = []
questions = expert_archive['questionText'].tolist()

for i in range(0, len(questions), BATCH_SIZE):
    batch = questions[i: i + BATCH_SIZE]
    vecs = st_model.encode(batch, convert_to_numpy=True).astype('float32')
    all_embeddings.extend(vecs)
    print(f"  Embedded {min(i + BATCH_SIZE, len(questions))}/{len(questions)}")

embeddings = np.array(all_embeddings).astype('float32')

# 5. Build and Save FAISS Index
os.makedirs("api/expert_archive", exist_ok=True)
index = faiss.IndexFlatL2(DIM)
index.add(embeddings)
faiss.write_index(index, "api/expert_archive/sentinel_brain.index")
expert_archive.to_pickle("api/expert_archive/expert_archive.pkl")
print(f"\nFAISS index saved: {index.ntotal} vectors @ {DIM}-dim")

# 6. Populate SQLite Database
print(f"Populating SQLite with {len(expert_archive)} records...")
db = SessionLocal()
try:
    db.query(ExpertBrainData).delete()
    for i, row in expert_archive.iterrows():
        db.add(ExpertBrainData(
            question=row['questionText'],
            answer=row['answerText'],
            source="CounselChat/MentalChat",
            embedding_id=i
        ))
        if i % 500 == 0:
            db.commit()
            print(f"  SQL: {i}/{len(expert_archive)}")
    db.commit()
    print("SQLite population complete.")
except Exception as e:
    print(f"SQL Error: {e}")
    db.rollback()
finally:
    db.close()

print("\n✅ Sentinel Expert Brain fully rebuilt.")
print(f"   FAISS: {index.ntotal} vectors | SQL: {len(expert_archive)} records")
