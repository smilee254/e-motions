import pandas as pd
df = pd.read_pickle("api/expert_archive/expert_archive.pkl")
print(f"Archive length: {len(df)}")
