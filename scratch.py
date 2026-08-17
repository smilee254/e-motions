import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    print("Testing gpt-oss-20b...")
    comp = client.chat.completions.create(
        messages=[{"role": "user", "content": "Return JSON {\"hello\": \"world\"}"}],
        model="openai/gpt-oss-20b",
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    print("20b success!")
except Exception as e:
    print(f"20b Error: {e}")

try:
    print("Testing gpt-oss-120b...")
    comp = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}],
        model="openai/gpt-oss-120b",
        temperature=0.6,
        max_tokens=300
    )
    print("120b success!")
except Exception as e:
    print(f"120b Error: {e}")
