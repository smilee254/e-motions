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

# 1. Pulling High-Fidelity Archives (CounselChat & MentalChat16K)
print("Downloading CounselChat & MentalChat archives...")
try:
    ds_counsel = load_dataset("nbertagnolli/counsel-chat")
    ds_mental = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
except Exception as e:
    print(f"Error downloading datasets: {e}")
    exit(1)

# 2. Extracting Expert Responses
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

# 3. Generating Semantic Embeddings
print(f"Embedding {len(expert_archive)} expert cases using Gemini...")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

embeddings = []
texts_to_embed = expert_archive['questionText'].tolist()

for i, text in enumerate(texts_to_embed):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(texts_to_embed)}")
    try:
        response = client.models.embed_content(
            model='models/gemini-embedding-001',
            contents=text,
        )
        if hasattr(response, 'embeddings'):
            embeddings.append(response.embeddings[0].values)
        else:
            embeddings.append(response['embedding'])
    except Exception as e:
        print(f"Error on index {i}: {e}")
        if "429" in str(e):
            time.sleep(60)
        embeddings.append([0.0] * 768)

embeddings = np.array(embeddings).astype('float32')

# 4. Save the Local FAISS Index
embeddings_f16 = embeddings.astype(np.float16)
index = faiss.IndexFlatL2(embeddings_f16.shape[1])
index.add(embeddings_f16.astype('float32'))

os.makedirs("api/expert_archive", exist_ok=True)
faiss.write_index(index, "api/expert_archive/sentinel_brain.index")
expert_archive.to_pickle("api/expert_archive/expert_archive.pkl")

# 5. Populate SQLite Database
print(f"Populating SQLite database with {len(expert_archive)} expert records...")
db = SessionLocal()
try:
    db.query(ExpertBrainData).delete()
    for i, row in expert_archive.iterrows():
        entry = ExpertBrainData(
            question=row['questionText'],
            answer=row['answerText'],
            source="CounselChat/MentalChat",
            embedding_id=i
        )
        db.add(entry)
    db.commit()
    print("Database population complete.")
except Exception as e:
    print(f"Error populating database: {e}")
    db.rollback()
finally:
    db.close()

print("Success! Sentinel Expert Brain is fully loaded.")
