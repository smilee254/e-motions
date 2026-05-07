import faiss
index = faiss.read_index("api/expert_archive/sentinel_brain.index")
print(f"Vectors in index: {index.ntotal}")
