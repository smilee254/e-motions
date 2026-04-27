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

# Regional Support Contacts (Kenyan Context)
REGIONAL_CONTACTS = {
    "Kiambu": "Kiambu Level 5 Hospital Mental Health Unit: 0722 000 000 (Mock)",
    "Nairobi": "Red Cross Nairobi Regional Branch: 1199",
    "Chuka": "Chuka University Counseling Center: 0711 000 000 (Mock)",
    "Mombasa": "Coast General Hospital Psychiatric Wing: 0733 000 000 (Mock)",
    "Nakuru": "Nakuru Level 5 Hospital: 0722 111 111 (Mock)",
    "Uasin Gishu": "Moi Teaching and Referral Hospital (MTRH): 053 2033041",
    "Kisumu": "Jaramogi Oginga Odinga Teaching & Referral Hospital: 057 2020130"
}

# Extended Kenyan Knowledge Base + Standard Peer Counseling
KNOWLEDGE_BASE = [
    # Kenyan Context & Success Stories
    {"input": "Sapa is hitting hard, I can't even afford fare.", "reply": "Financial pressure can make the world feel very small. Like Kamau from Kiambu who started with just 50 bob, sometimes the smallest steps are the most resilient ones. What's one thing we can focus on today?"},
    {"input": "Black tax is too much", "reply": "The weight of being the 'provider' is heavy. Many have found that setting 'soft boundaries' with family is a survival skill, not a betrayal. It's okay to breathe for yourself too."},
    {"input": "I failed my exams at Chuka", "reply": "Chuka is tough, but remember Sarah who failed her first year but eventually built a tech startup in Meru? This unit doesn't define your capacity to create. Let's look at the next step."},
    
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
    # Use local references for type safety
    e, i = embedder, index
    if e is None or i is None:
        return "I am here and I'm listening. Your thoughts are safe in this sanctuary."
        
    try:
        query_vector = e.encode([user_text])
        # Find the top 1 closest match
        D, I = i.search(query_vector.astype('float32'), 1)
        match_idx = I[0][0]
        
        # If no good match found (index safety), return default
        if match_idx == -1 or match_idx >= len(KNOWLEDGE_BASE):
            return "I hear you. Tell me more about what's on your mind."
            
        return KNOWLEDGE_BASE[match_idx]["reply"]
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        return "The sanctuary is a quiet space for your thoughts. I'm with you."

def get_regional_grounding(region: str) -> str:
    """Provides local context for the AI."""
    contact = REGIONAL_CONTACTS.get(region, "the nearest Red Cross branch")
    return f"Since you're in {region}, I'm keeping an eye on local resources for you. If things feel too heavy, {contact} is available."

def detect_depth(text: str) -> float:
    """Returns a depth score from 0.0 to 1.0 based on keyword intensity."""
    heavy_terms = ["black tax", "sapa", "suicide", "die", "give up", "hopeless", "broken", "failed"]
    score = sum(0.2 for term in heavy_terms if term in text.lower())
    return min(score, 1.0)
