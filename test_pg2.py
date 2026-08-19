import os, asyncio
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq()
print("Safe:", client.chat.completions.create(
    model='meta-llama/llama-prompt-guard-2-86m', 
    messages=[{'role': 'user', 'content': 'I am feeling really sad today, can we talk?'}], 
    temperature=0).choices[0].message.content)
