from datasets import load_dataset
import pandas as pd

try:
    ds_mental = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
    print("Mental Health Dataset Columns:", ds_mental['train'].column_names)
    print("First 5 rows:")
    print(ds_mental['train'][:5])
except Exception as e:
    print(f"Error: {e}")
