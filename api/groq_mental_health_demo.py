import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from google import genai

# Load environment variables (.env file where API keys are stored)
load_dotenv()

# ==========================================
# 1. INITIALIZE CLIENTS (The RAG Stack)
# ==========================================

# A. Groq: Ultra-fast LLM generation (using Llama 3)
# Requires GROQ_API_KEY in .env
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# B. Gemini: Used only for creating text embeddings to match your database
# Requires GEMINI_API_KEY in .env
google_api_key = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=google_api_key) if google_api_key else None
EMBED_MODEL = "models/gemini-embedding-001"

# C. Qdrant: Vector database hosting your Expert Archive (CounselChat/MentalChat)
# Requires QDRANT_URL and QDRANT_API_KEY in .env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sentinel_brain")

qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

# ==========================================
# 2. SYSTEM PROMPT (The Persona)
# ==========================================
SYSTEM_PROMPT = """
You are Sentinel, a grounded, empathetic peer in the e-motions sanctuary.
Your wisdom is backed by expert archives, but your voice is human, casual, and supportive.

Guidelines for your responses:
1. Validation First: Always mirror and acknowledge the user's feelings before offering advice.
2. Integrate Expert Wisdom: When 'EXPERT CONTEXT' is provided, let it inform your response, but NEVER quote it directly or sound like a textbook. Translate the wisdom into a casual, peer-to-peer tone (e.g., "I read somewhere that...", or just weaving the advice in naturally).
3. Keep it Conversational: Respond like a caring friend in a chat. Avoid long bulleted lists or essays.
4. No Diagnostics: You are a supportive AI, not a doctor.

🚨 EMERGENCY PROTOCOL:
If the user mentions self-harm, suicide, or intent to harm others, immediately break character, validate their pain, and provide the Red Cross contact: 1199 (Kenya) or emergency services.
"""

# ==========================================
# 3. CORE RAG LOGIC
# ==========================================

def embed_message(message: str) -> list:
    """Uses Gemini API to turn the user's message into vector coordinates."""
    if not ai_client:
        return []
    try:
        response = ai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=message[:2000],
        )
        if hasattr(response, 'embeddings') and response.embeddings:
            return response.embeddings[0].values
        elif hasattr(response, 'embedding'):
            return response.embedding.values
    except Exception as e:
        print(f"[!] Embedding error: {e}")
    return []

def get_expert_context(message: str) -> str:
    """Searches Qdrant for previously embedded mental health expert responses."""
    if not qdrant_client:
        return ""
    
    query_vec = embed_message(message)
    if not query_vec:
        return ""
        
    context = ""
    try:
        # We look for the top 2 closest expert responses in your Qdrant DB
        results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vec,
            limit=2,
            score_threshold=0.5, # Only confident matches
        )
        for hit in results.points:
            answer = hit.payload.get("answer", "")
            if answer and answer not in context:
                context += f"\n- {answer[:400]}..."
    except Exception as e:
        print(f"[!] Qdrant query error: {e}")
        
    return context

def generate_response(user_message: str, chat_history: list) -> str:
    """The main pipeline: Retrieve -> Build Prompt -> Generate via Groq."""
    
    # 1. Retrieval: Fetch relevant expert wisdom based on user input
    print("   [Retrieving expert context...]")
    expert_context = get_expert_context(user_message)
    
    # 2. Augmentation: Combine Persona + History + Context
    dynamic_system_prompt = f"{SYSTEM_PROMPT}\n"
    if expert_context:
        dynamic_system_prompt += f"\nEXPERT CONTEXT (Use this to guide your response discretely):\n{expert_context}\n"
        
    messages = [{"role": "system", "content": dynamic_system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})
    
    # 3. Generation: Get lightning-fast response from Groq
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-120b",
            temperature=0.6,
            max_tokens=300,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq Generation Error: {e}"

# ==========================================
# 4. INTERACTIVE DEMO LOOP
# ==========================================
if __name__ == "__main__":
    print("=========================================================")
    print("🧠 Sentinel RAG Demo: Gemini (Embed) + Qdrant (DB) + Groq (Brain)")
    print("=========================================================")
    
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️ WARNING: GROQ_API_KEY is not set in your .env file!")
        print("Get one at: https://console.groq.com/\n")
        
    history = []
    while True:
        msg = input("\nYou: ")
        if msg.lower() in ['quit', 'exit']:
            break
            
        response = generate_response(msg, history)
        
        print(f"\nSentinel: {response}")
        
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response})
