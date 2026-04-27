from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

# Configure logging
logger = logging.getLogger("e-motions-fallback")

# Load the lightweight model (approx 80MB)
# This will download on first run
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load sentence-transformer: {e}")
    embedder = None

# Extended Kenyan Knowledge Base + Standard Peer Counseling
KNOWLEDGE_BASE = [
    # Kenyan Context
    {"input": "Sapa is hitting hard, I can't even afford fare.", "reply": "Financial pressure can make the world feel very small and suffocating. You aren't alone in this hustle. What's one thing we can focus on that doesn't cost a cent today?"},
    {"input": "My siblings expect me to pay for everything.", "reply": "That weight of being the 'provider' is heavy, especially when you're still trying to find your own feet. It’s okay to feel overwhelmed by those expectations."},
    {"input": "These Actuarial units are going to fail me.", "reply": "Calculus and Bivariate Probability can feel like a mountain. Don't let the grades define your worth as a builder. How about we break the mountain into smaller stones?"},
    {"input": "My friends left me out of their business.", "reply": "Feeling like a 'third wheel' or being excluded by your circle is a deep sting. This sanctuary is a space where you don't have to compete for a seat at the table."},
    {"input": "My parents don't understand my career path.", "reply": "There is a massive gap sometimes between what they want for us and who we actually are. It takes a lot of courage to keep building your own path despite that."},
    {"input": "I'm just tired of being alone in this room.", "reply": "The silence in a room can get very loud when you're struggling. I am here, and I am listening. Tell me about the last thing that made you feel even a little bit like yourself."},
    
    # Slang / Informal
    {"input": "Sapa is killing me", "reply": "Financial stress in this economy is a real weight. You're doing your best with what you have. Let's talk through the heavy parts."},
    {"input": "Black tax is too much", "reply": "The pressure to provide for everyone can make you feel like you're disappearing. It's okay to acknowledge how exhausting that is."},
    {"input": "I failed my exams at Chuka", "reply": "One paper doesn't define a builder. You've got the skills to create systems; don't let the theory struggles dim that light."},
    
    # Standard Peer Support
    {"input": "I feel like a burden", "reply": "It sounds like you're carrying a lot right now. Sharing isn't a burden; it's the first step to lightening the load."},
    {"input": "I can't sleep because of stress", "reply": "Nights can be the hardest when the world is quiet but your mind isn't. What's one thing you'd like to 'park' here so you can rest?"},
    {"input": "Nobody cares about me", "reply": "That is a heavy and painful feeling to hold. I am here, and I am listening. Tell me more about what's making you feel unseen."},
    {"input": "I want to give up", "reply": "I hear how exhausted you are. It's okay to feel like you've reached your limit. I'm here to sit with you in that space for as long as you need."},
    {"input": "Everything is falling apart", "reply": "When everything feels chaotic, it's hard to find a place to stand. Let's take it one small breath at a time. What's the very next thing in front of you?"}
]

# Build the Vector Index
if embedder:
    try:
        sentences = [item["input"] for item in KNOWLEDGE_BASE]
        encoded_data = embedder.encode(sentences)
        index = faiss.IndexFlatL2(encoded_data.shape[1])
        index.add(encoded_data.astype('float32'))
    except Exception as e:
        logger.error(f"Failed to build faiss index: {e}")
        index = None
else:
    index = None

def get_kenyan_fallback(user_text):
    """
    Finds the closest matching expert response using semantic search.
    If the index or embedder is missing, returns a generic compassionate response.
    """
    if not embedder or not index:
        return "I am here and I'm listening. Your thoughts are safe in this sanctuary."
        
    try:
        query_vector = embedder.encode([user_text])
        # Find the top 1 closest match
        D, I = index.search(query_vector.astype('float32'), 1)
        match_idx = I[0][0]
        
        # If no good match found (index safety), return default
        if match_idx == -1 or match_idx >= len(KNOWLEDGE_BASE):
            return "I hear you. Tell me more about what's on your mind."
            
        return KNOWLEDGE_BASE[match_idx]["reply"]
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        return "The sanctuary is a quiet space for your thoughts. I'm with you."
