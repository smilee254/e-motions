import pickle
import pandas as pd
with open("api/expert_archive/expert_archive.pkl", "rb") as f:
    try:
        data = pickle.load(f)
        print(f"Data type: {type(data)}")
        if isinstance(data, pd.DataFrame):
             print(f"Columns: {data.columns}")
             print(f"Length: {len(data)}")
    except Exception as e:
        print(f"Error: {e}")
