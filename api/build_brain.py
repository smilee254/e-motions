import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import os

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
# Using 'all-MiniLM-L6-v2' for performance on your Celeron processor
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Embedding {len(expert_archive)} expert cases (1.7M+ tokens)...")
embeddings = model.encode(expert_archive['questionText'].tolist(), show_progress_bar=True)

# 4. Save the Local FAISS Index (Zero-Quota Retrieval)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))

# Ensure api directory exists
os.makedirs("api", exist_ok=True)

faiss.write_index(index, "api/sentinel_brain.index")
expert_archive.to_pickle("api/expert_archive.pkl")

print("Success! Sentinel is now loaded with expanded expert counseling archives.")
