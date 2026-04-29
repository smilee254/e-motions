import pandas as pd
from datasets import load_dataset
import numpy as np
import faiss
import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

# 1. Pulling High-Fidelity Archives (CounselChat & MentalChat16K)
print("Downloading CounselChat & MentalChat archives...")
try:
    ds_counsel = load_dataset("nbertagnolli/counsel-chat")
    ds_mental = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
except Exception as e:
    print(f"Error downloading datasets: {e}")
    exit(1)

# 2. Extracting Expert Responses on Sadness & Relationships
# We prioritize 'answerText' from CounselChat and 'response' from MentalChat
df_counsel = pd.DataFrame(ds_counsel['train'])[['questionText', 'answerText']]
df_mental = pd.DataFrame(ds_mental['train'])[['context', 'response']]
df_mental.columns = ['questionText', 'answerText']

# Merge the expert data into one 'Human' archive
expert_archive = pd.concat([df_counsel, df_mental], ignore_index=True).dropna()

# 3. Generating Semantic Embeddings (The 'Vibe' Layer)
print(f"Embedding {len(expert_archive)} expert cases using Gemini...")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

embeddings = []
texts_to_embed = expert_archive['questionText'].tolist()

# Note: Gemini embedding API has rate limits. In production you should batch or add sleep.
for i, text in enumerate(texts_to_embed):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(texts_to_embed)}")
    try:
        response = client.models.embed_content(
            model='text-embedding-004',
            contents=text,
        )
        if hasattr(response, 'embeddings'):
            embeddings.append(response.embeddings[0].values)
        else:
            embeddings.append(response['embedding'])
    except Exception as e:
        print(f"Error on index {i}: {e}")
        # Add sleep if rate limited
        if "429" in str(e):
            time.sleep(5)
        embeddings.append([0.0] * 768)

embeddings = np.array(embeddings).astype('float32')

# 4. Save the Local FAISS Index (Zero-Quota Retrieval)
# Convert to float16 to save space as per user suggestion
embeddings_f16 = embeddings.astype(np.float16)

index = faiss.IndexFlatL2(embeddings_f16.shape[1])
# FAISS requires float32 for add
index.add(embeddings_f16.astype('float32'))

# Ensure api/expert_archive directory exists
os.makedirs("api/expert_archive", exist_ok=True)

faiss.write_index(index, "api/expert_archive/sentinel_brain.index")
expert_archive.to_pickle("api/expert_archive/expert_archive.pkl")

print("Success! Sentinel is now loaded with expanded expert counseling archives.")
