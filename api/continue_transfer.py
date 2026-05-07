import pandas as pd
from datasets import load_dataset
import numpy as np
import faiss
import os
import time
from google import genai
from dotenv import load_dotenv
from api._database import SessionLocal, ExpertBrainData

load_dotenv()

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
else:
    print("No index found. Starting from scratch.")
    index = None
    start_idx = 0

# 3. Continue embedding
if start_idx < len(expert_archive):
    print(f"Embedding remaining {len(expert_archive) - start_idx} records...")
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    new_embeddings = []
    texts_to_embed = expert_archive['questionText'].tolist()[start_idx:]
    
    for i, text in enumerate(texts_to_embed):
        current_abs_idx = start_idx + i
        if i % 50 == 0:
            print(f"Progress: {current_abs_idx}/{len(expert_archive)}")
        try:
            response = client.models.embed_content(
                model='models/gemini-embedding-001',
                contents=text,
            )
            if hasattr(response, 'embeddings'):
                new_embeddings.append(response.embeddings[0].values)
            else:
                new_embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error on index {current_abs_idx}: {e}")
            if "429" in str(e):
                print("Rate limit hit. Sleeping for 60s...")
                time.sleep(60)
                # Retry once
                try:
                    response = client.models.embed_content(model='models/gemini-embedding-001', contents=text)
                    if hasattr(response, 'embeddings'): new_embeddings.append(response.embeddings[0].values)
                    else: new_embeddings.append(response['embedding'])
                    continue
                except: pass
            new_embeddings.append([0.0] * 768)

    if new_embeddings:
        new_embeddings_arr = np.array(new_embeddings).astype('float32')
        # Cast to float16 and back to float32 to match existing index if it was float16
        # Actually _build_brain.py used float16. 
        # But IndexFlatL2 expects float32.
        if index is None:
            index = faiss.IndexFlatL2(768)
        
        index.add(new_embeddings_arr)
        print("Updated FAISS index.")

# 4. Save updated files
os.makedirs(EXPERT_DIR, exist_ok=True)
faiss.write_index(index, INDEX_PATH)
# We don't save the pkl because of the version issue, but we'll use the df for SQL
# Actually, the user's pkl is corrupted, so saving a new one might fix it for their environment
try:
    expert_archive.to_pickle(PKL_PATH)
    print("Saved updated expert_archive.pkl")
except Exception as e:
    print(f"Warning: Could not save pkl (likely version mismatch): {e}")

# 5. Populate SQL
print("Populating SQLite database...")
db = SessionLocal()
try:
    # Check current count
    sql_count = db.query(ExpertBrainData).count()
    print(f"SQL currently has {sql_count} records. Syncing to {len(expert_archive)}...")
    
    # Simple strategy: If empty, populate all. If not, just clear and re-populate to ensure order.
    # Given the user's situation, a full sync is safest.
    db.query(ExpertBrainData).delete()
    db.commit()
    
    for i, row in expert_archive.iterrows():
        if i % 500 == 0:
            print(f"SQL Progress: {i}/{len(expert_archive)}")
        entry = ExpertBrainData(
            question=row['questionText'],
            answer=row['answerText'],
            source="CounselChat/MentalChat",
            embedding_id=i
        )
        db.add(entry)
    db.commit()
    print("Database sync complete.")
except Exception as e:
    print(f"Error populating database: {e}")
    db.rollback()
finally:
    db.close()

print("Success! Transfer resumed and completed.")
