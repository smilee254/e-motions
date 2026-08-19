"""
Sentinel → Qdrant Custom Dataset Upload Script
==============================================
Uploads our custom Kenyan mental health scenarios to Qdrant Cloud.
"""

import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "sentinel_brain")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 3072
CSV_PATH = "data/kenyan_context.csv"

print("Connecting to Qdrant Cloud...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
print("Connected ✓")

collections = [c.name for c in qdrant.get_collections().collections]
if COLLECTION not in collections:
    print(f"Creating collection '{COLLECTION}'...")
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print("Collection created ✓")

gemini_client = genai.Client(api_key=GEMINI_KEY)

def embed_batch(texts: list) -> list:
    for attempt in range(5):
        try:
            response = gemini_client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
            )
            if hasattr(response, 'embeddings') and response.embeddings:
                return [e.values for e in response.embeddings]
            else:
                return [response.embedding.values]
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  Rate limited. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"  Embedding error: {e}")
                time.sleep(5)
    raise RuntimeError("Failed to embed after retries")

def main():
    print(f"\nLoading custom scenarios from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, escapechar='\\', on_bad_lines='skip', engine='python')
    print(f"Loaded {len(df)} records")

    questions = df['Question'].tolist()
    answers = df['Answer'].tolist()

    print("Embedding scenarios with Gemini...")
    vectors = embed_batch(questions)

    points = []
    for i, (vec, q, a) in enumerate(zip(vectors, questions, answers)):
        # Use a large ID range for custom data to avoid overwriting existing data
        point_id = 90000 + i 
        points.append(PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "question": str(q),
                "answer": str(a),
                "source": "Kenyan Context - Custom",
                "embedding_id": point_id,
            }
        ))

    print("Uploading to Qdrant...")
    qdrant.upsert(collection_name=COLLECTION, points=points)
    
    info = qdrant.get_collection(COLLECTION)
    print(f"\n✅ Upload complete! Total vectors in '{COLLECTION}': {info.points_count}")

if __name__ == "__main__":
    main()
