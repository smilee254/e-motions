import pandas as pd
from datasets import load_dataset
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from api._database import SessionLocal, ExpertBrainData

print("Loading local sentence-transformer embedder (all-MiniLM-L6-v2)...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Embedder ready. Dimension: {st_model.get_sentence_embedding_dimension()}")

# 1. Re-assemble the archive from datasets
print("Re-downloading datasets to match index...")
try:
    ds_counsel = load_dataset("nbertagnolli/counsel-chat")
    ds_mental = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
except Exception as e:
    print(f"Error downloading datasets: {e}")
    exit(1)

# Extract expert responses (same logic as _build_brain.py)
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
print(f"Total target records: {len(expert_archive)}")

# 2. Load existing index
EXPERT_DIR = "api/expert_archive"
INDEX_PATH = os.path.join(EXPERT_DIR, "sentinel_brain.index")
PKL_PATH = os.path.join(EXPERT_DIR, "expert_archive.pkl")

if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    start_idx = index.ntotal
    print(f"Starting from index {start_idx}/{len(expert_archive)}")
    
    # Ensure SQL is caught up
    db = SessionLocal()
    sql_count = db.query(ExpertBrainData).count()
    if sql_count < start_idx:
        print(f"SQL is behind ({sql_count} < {start_idx}). Catching up...")
        for i in range(sql_count, start_idx):
            if i % 500 == 0:
                print(f"SQL Catch-up: {i}/{start_idx}")
            row = expert_archive.iloc[i]
            entry = ExpertBrainData(
                question=row['questionText'],
                answer=row['answerText'],
                source="CounselChat/MentalChat",
                embedding_id=i
            )
            db.add(entry)
        db.commit()
        print("SQL caught up.")
    db.close()
else:
    print("No index found. Starting from scratch.")
    index = None
    start_idx = 0

# 3. Continue embedding with sentence-transformers (no rate limits, 384-dim matches FAISS index)
if start_idx < len(expert_archive):
    remaining = expert_archive['questionText'].tolist()[start_idx:]
    print(f"Embedding {len(remaining)} remaining records with sentence-transformers...")

    BATCH_SIZE = 64  # sentence-transformers handles batches efficiently
    new_embeddings = []

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_texts = remaining[batch_start: batch_start + BATCH_SIZE]
        batch_vecs = st_model.encode(batch_texts, convert_to_numpy=True).astype('float32')
        new_embeddings.extend(batch_vecs)

        # Save progress after each batch
        abs_end_idx = start_idx + batch_start + len(batch_texts) - 1
        new_embeddings_arr = np.array(new_embeddings).astype('float32')

        if index is None:
            index = faiss.IndexFlatL2(new_embeddings_arr.shape[1])
        index.add(new_embeddings_arr)
        faiss.write_index(index, INDEX_PATH)

        # Sync SQL for this batch
        db = SessionLocal()
        try:
            for j, _ in enumerate(batch_vecs):
                row_idx = start_idx + batch_start + j
                row = expert_archive.iloc[row_idx]
                existing = db.query(ExpertBrainData).filter(
                    ExpertBrainData.embedding_id == row_idx
                ).first()
                if not existing:
                    db.add(ExpertBrainData(
                        question=row['questionText'],
                        answer=row['answerText'],
                        source="CounselChat/MentalChat",
                        embedding_id=row_idx
                    ))
            db.commit()
            print(f"  Progress: {abs_end_idx + 1}/{len(expert_archive)} records")
        except Exception as sql_e:
            print(f"SQL Error: {sql_e}")
            db.rollback()
        finally:
            db.close()

        new_embeddings = []  # Reset for next batch (already added to FAISS)

# 4. Final Save & Metadata
print("Ensuring metadata consistency...")
try:
    expert_archive.to_pickle(PKL_PATH)
    print("Saved updated expert_archive.pkl")
except Exception as e:
    print(f"Warning: Could not save pkl: {e}")

# Verify final counts
db = SessionLocal()
sql_count = db.query(ExpertBrainData).count()
index_count = index.ntotal if index else 0
db.close()

print(f"\nFinal Status:")
print(f"FAISS Vectors: {index_count}")
print(f"SQL Records: {sql_count}")
print(f"Target Total: {len(expert_archive)}")

if index_count == len(expert_archive) and sql_count == len(expert_archive):
    print("✅ SUCCESS: Expert brain is fully synchronized and populated.")
else:
    print("⚠️ INCOMPLETE: Run the script again later to finish remaining records.")
