"""
Sentinel → Qdrant Cloud Upload Script
======================================
Embeds all 2784 expert archive records using Gemini text-embedding-004
(768-dim, no local ML model, no PyTorch) and uploads them to Qdrant Cloud.

Run once:
    PYTHONPATH=. python3 api/_upload_to_qdrant.py

Resume from a checkpoint if interrupted:
    PYTHONPATH=. python3 api/_upload_to_qdrant.py --resume
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# --- Config ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "sentinel_brain")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "models/gemini-embedding-001"  # 3072-dim, available on v1beta
EMBED_DIM = 3072
BATCH_SIZE = 50                       # texts per Gemini batch call
CHECKPOINT_FILE = "api/expert_archive/qdrant_upload_checkpoint.json"
PKL_PATH = "api/expert_archive/expert_archive.pkl"

# --- Init clients ---
print("Connecting to Qdrant Cloud...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
print("Connected ✓")

gemini_client = genai.Client(api_key=GEMINI_KEY)

# --- Ensure collection exists ---
collections = [c.name for c in qdrant.get_collections().collections]
if COLLECTION not in collections:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"Created Qdrant collection '{COLLECTION}' ({EMBED_DIM}-dim, cosine) ✓")
else:
    info = qdrant.get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' exists — {info.points_count} points already stored")


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f).get("last_uploaded_idx", -1)
    return -1


def save_checkpoint(idx: int):
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_uploaded_idx": idx}, f)


def embed_single(text: str) -> list[float]:
    """Embed a single text string using Gemini gemini-embedding-001."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
            )
            # Response has .embeddings list
            if hasattr(response, 'embeddings') and response.embeddings:
                return response.embeddings[0].values
            elif hasattr(response, 'embedding'):
                return response.embedding.values
            else:
                raise ValueError(f"Unexpected response shape: {response}")
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"  Embedding error: {e}")
                time.sleep(5)
    raise RuntimeError("Failed to embed text after all retries")


def embed_batch(texts: list) -> list:
    """Embed a batch one-by-one (Gemini free tier doesn't support true batching)."""
    vectors = []
    for text in texts:
        vec = embed_single(str(text)[:2000])  # cap at 2000 chars
        vectors.append(vec)
        time.sleep(0.1)  # small delay between calls
    return vectors


def main(resume: bool = False):
    # Load archive
    print(f"\nLoading expert archive from {PKL_PATH}...")
    df = pd.read_pickle(PKL_PATH)
    print(f"Loaded {len(df)} records")

    start_idx = 0
    if resume:
        checkpoint = load_checkpoint()
        start_idx = checkpoint + 1
        print(f"Resuming from index {start_idx}")

    if start_idx >= len(df):
        print("All records already uploaded. Done!")
        return

    remaining = len(df) - start_idx
    total_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Uploading {remaining} records in {total_batches} batches of {BATCH_SIZE}...\n")

    for batch_num, batch_start in enumerate(range(start_idx, len(df), BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        questions = batch_df['questionText'].tolist()
        answers = batch_df['answerText'].tolist()

        # Embed questions
        try:
            vectors = embed_batch(questions)
        except RuntimeError as e:
            print(f"  FATAL: {e}. Saving checkpoint at {batch_start - 1} and exiting.")
            save_checkpoint(batch_start - 1)
            sys.exit(1)

        # Build Qdrant points
        points = []
        for j, (vec, question, answer) in enumerate(zip(vectors, questions, answers)):
            global_idx = batch_start + j
            points.append(PointStruct(
                id=global_idx,
                vector=vec,
                payload={
                    "question": str(question)[:500],
                    "answer": str(answer)[:1000],
                    "source": "CounselChat/MentalChat",
                    "embedding_id": global_idx,
                }
            ))

        # Upsert to Qdrant
        qdrant.upsert(collection_name=COLLECTION, points=points)
        save_checkpoint(batch_end - 1)

        pct = (batch_end / len(df)) * 100
        print(f"  [{batch_end}/{len(df)}] {pct:.0f}% — batch {batch_num + 1}/{total_batches} ✓")

        # Small delay to respect Gemini rate limits
        time.sleep(0.5)

    # Final stats
    info = qdrant.get_collection(COLLECTION)
    print(f"\n✅ Upload complete! Qdrant collection '{COLLECTION}': {info.points_count} vectors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)
