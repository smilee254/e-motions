from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import logging

# Configure logging
logger = logging.getLogger("e-motions-fallback")

# Load the lightweight model (approx 80MB)
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

# --- Sentiment Guard ---
# If the message is positive, never match it to crisis responses
POSITIVE_SIGNALS = [
    "happy", "great", "good", "excited", "amazing", "blessed",
    "grateful", "fine", "okay", "okay", "chillin", "vibing",
    "better", "relieved", "calm", "peaceful", "joy", "love"
]

POSITIVE_RESPONSES = [
    "That's good to hear! What's been the highlight of your day so far?",
    "Love that energy! What's making things feel good right now?",
    "Glad to hear you're in a good space. Anything specific lifting your vibe today?",
    "Nice! It's always good when things are flowing. How long has it been feeling this way?",
    "That's what the sanctuary likes to hear. What's been going right for you?"
]

# Extended Kenyan Knowledge Base — covers both positive and support scenarios
KNOWLEDGE_BASE = [
    # Positive / Neutral
    {"input": "I feel happy", "reply": "That's good to hear! What's been making things feel good?"},
    {"input": "I'm doing great today", "reply": "Love that energy! What's lifting your vibe today?"},
    {"input": "Things are going well", "reply": "Nice! It's good when life flows like that. What's been going right?"},

    # Kenyan Context
    {"input": "Sapa is hitting hard, I can't even afford fare.", "reply": "Financial pressure can make the world feel very small. Like Kamau from Kiambu who started with just 50 bob, sometimes the smallest steps are the most resilient ones. What's one thing we can focus on today?"},
    {"input": "Black tax is too much", "reply": "The weight of being the 'provider' is heavy. Many have found that setting 'soft boundaries' with family is a survival skill, not a betrayal. It's okay to breathe for yourself too."},
    {"input": "I failed my exams at Chuka", "reply": "Chuka is tough, but remember Sarah who failed her first year but eventually built a tech startup in Meru? This unit doesn't define your capacity to create. Let's look at the next step."},

    # Support
    {"input": "I feel like a burden", "reply": "It sounds like you're carrying a lot right now. Sharing isn't a burden; it's the first step to lightening the load."},
    {"input": "I can't sleep because of stress", "reply": "Nights can be the hardest when the world is quiet but your mind isn't. What's one thing you'd like to 'park' here so you can rest?"},
    {"input": "Nobody cares about me", "reply": "That is a heavy and painful feeling to hold. I am here, and I am listening. Tell me more about what's making you feel unseen."},
    {"input": "I want to give up", "reply": "I hear how exhausted you are. It's okay to feel like you've reached your limit. I'm here to sit with you in that space for as long as you need."},
    {"input": "Everything is falling apart", "reply": "When everything feels chaotic, it's hard to find a place to stand. Let's take it one small breath at a time. What's the very next thing in front of you?"}
]

# Build the local Vector Index from KNOWLEDGE_BASE
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

# --- Expert Brain (CounselChat + MentalChat16K) ---
# Loaded by main.py at startup — if available, used instead of tiny local index
_expert_index = None
_expert_archive = None

def load_expert_brain_refs(exp_index, exp_archive):
    """Called from main.py to share the expert brain references into fallback."""
    global _expert_index, _expert_archive
    _expert_index = exp_index
    _expert_archive = exp_archive

def get_kenyan_fallback(user_text: str) -> str:
    """
    Intent-aware fallback:
    1. Positive sentiment → return uplifting response (never a crisis reply)
    2. Use Expert Brain (CounselChat) if available
    3. Fall back to local KNOWLEDGE_BASE FAISS search
    """
    user_lower = user_text.lower()

    # 1. Positivity Guard — short-circuit before any FAISS search
    if any(word in user_lower for word in POSITIVE_SIGNALS):
        import random
        return random.choice(POSITIVE_RESPONSES)

    # 2. Try Expert Brain first (CounselChat + MentalChat16K)
    e = embedder
    if e is not None and _expert_index is not None and _expert_archive is not None:
        try:
            query_vec = e.encode([user_text])
            D, I = _expert_index.search(query_vec.astype('float32'), 1)
            match_idx = I[0][0]
            if match_idx != -1 and match_idx < len(_expert_archive):
                raw = _expert_archive.iloc[match_idx]['answerText']
                # Trim to a conversational length
                raw_str = str(raw).strip()
                if len(raw_str) > 280:
                    return raw_str[:280] + "..."
                return raw_str
        except Exception as ex:
            logger.error(f"Expert brain fallback error: {ex}")

    # 3. Local KNOWLEDGE_BASE FAISS search
    local_e = embedder
    local_i = index
    if local_e is None or local_i is None:
        return "I am here and I'm listening. Your thoughts are safe in this sanctuary."

    try:
        query_vector = local_e.encode([user_text])
        D, I = local_i.search(query_vector.astype('float32'), 1)
        match_idx = I[0][0]

        # Distance threshold — if match is too far, use a generic response
        if match_idx == -1 or match_idx >= len(KNOWLEDGE_BASE) or D[0][0] > 1.5:
            return "I hear you. Take your time — what's on your mind right now?"

        return KNOWLEDGE_BASE[match_idx]["reply"]
    except Exception as ex:
        logger.error(f"Fallback search error: {ex}")
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
