from datasets import load_dataset
ds_mental = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
print(f"Mental Health Dataset Size: {len(ds_mental['train'])}")
ds_counsel = load_dataset("nbertagnolli/counsel-chat")
print(f"Counsel Chat Dataset Size: {len(ds_counsel['train'])}")
