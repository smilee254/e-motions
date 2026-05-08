import os
import numpy as np
import faiss
import logging

# Configure logging
logger = logging.getLogger("e-motions-fallback")


class LocalEmbedder:
    """
    Lazy-loading sentence-transformer embedder.
    - Model loads on first encode() call, NOT at import time (saves ~200MB at startup)
    - Returns None if sentence-transformers is not installed (Render free tier)
    - Callers must check for None before using the result
    """
    def __init__(self):
        self._model = None
        self._available = None  # None=unknown, True=ok, False=not installed

    def _load(self):
        if self._available is False:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self._available = True
                logger.info("LocalEmbedder loaded on first use (384-dim).")
            except ImportError:
                self._available = False
                logger.info("sentence-transformers not installed — FAISS disabled, SQL search active.")
                return None
            except Exception as e:
                self._available = False
                logger.error(f"Failed to load LocalEmbedder: {e}")
                return None
        return self._model

    def encode(self, texts):
        """Returns float32 ndarray, or None if the embedder is unavailable."""
        model = self._load()
        if model is None:
            return None
        try:
            return model.encode(texts, convert_to_numpy=True).astype('float32')
        except Exception as e:
            logger.error(f"LocalEmbedder encode error: {e}")
            return None


embedder = LocalEmbedder()

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
POSITIVE_SIGNALS = [
    "happy", "great", "good", "excited", "amazing", "blessed",
    "grateful", "fine", "okay", "chillin", "vibing",
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

# --- Expert Brain (CounselChat + MentalChat16K) ---
# Loaded by index.py at startup — if available, used for semantic search
_expert_index = None
_expert_archive = None

def load_expert_brain_refs(exp_index, exp_archive):
    """Called from index.py to share the expert brain references into fallback."""
    global _expert_index, _expert_archive
    _expert_index = exp_index
    _expert_archive = exp_archive


def _kb_keyword_match(user_lower: str) -> str | None:
    """
    Lightweight keyword match against the 10-entry local KNOWLEDGE_BASE.
    No FAISS index needed for a dataset this small — avoids startup RAM cost.
    """
    for item in KNOWLEDGE_BASE:
        key_words = [w for w in item["input"].lower().split() if len(w) > 3]
        hits = sum(1 for w in key_words if w in user_lower)
        if hits >= 2:
            return item["reply"]
    return None


def get_kenyan_fallback(user_text: str) -> str:
    """
    Intent-aware fallback (used only when Gemini is unavailable):
    1. Positive sentiment (no negation) → return uplifting response
    2. Expert Brain (CounselChat FAISS) if loaded → semantic search
    3. Local KNOWLEDGE_BASE keyword match → small curated replies
    4. Generic holding response
    """
    user_lower = user_text.lower()

    # 1. Positivity Guard — skip if negation words present
    # Prevents "not happy" from triggering a positive response
    _negations = ["not ", "never ", "don't ", "can't ", "won't ", "isn't ", "aren't ", "wasn't ", "no "]
    _has_negation = any(neg in user_lower for neg in _negations)
    if not _has_negation and any(word in user_lower for word in POSITIVE_SIGNALS):
        import random
        return random.choice(POSITIVE_RESPONSES)

    # 2. Try Expert Brain (CounselChat + MentalChat16K) — FAISS semantic search
    if _expert_index is not None and _expert_archive is not None:
        try:
            query_vec = embedder.encode([user_text])
            if query_vec is None:
                raise Exception("Embedder unavailable")
            D, I = _expert_index.search(query_vec.astype('float32'), 1)
            match_idx = I[0][0]
            if match_idx != -1 and match_idx < len(_expert_archive):
                raw_str = str(_expert_archive.iloc[match_idx]['answerText']).strip()
                if len(raw_str) > 280:
                    return raw_str[:280] + "..."
                return raw_str
        except Exception as ex:
            logger.error(f"Expert brain fallback error: {ex}")

    # 3. Local KNOWLEDGE_BASE keyword match (no FAISS, no model load)
    kb_reply = _kb_keyword_match(user_lower)
    if kb_reply:
        return kb_reply

    # 4. Generic holding response
    return "I hear you. Take your time — what's on your mind right now?"


def get_regional_grounding(region: str) -> str:
    """Provides local context for the AI."""
    contact = REGIONAL_CONTACTS.get(region, "the nearest Red Cross branch")
    return f"Since you're in {region}, I'm keeping an eye on local resources for you. If things feel too heavy, {contact} is available."

def detect_depth(text: str) -> float:
    """Returns a depth score from 0.0 to 1.0 based on keyword intensity."""
    heavy_terms = ["black tax", "sapa", "suicide", "die", "give up", "hopeless", "broken", "failed"]
    score = sum(0.2 for term in heavy_terms if term in text.lower())
    return min(score, 1.0)
