import os
import logging
import random
import re

# Configure logging
logger = logging.getLogger("e-motions-fallback")

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
    {"input": "I feel happy", "reply": "That's good to hear! What's been making things feel good?"},
    {"input": "I'm doing great today", "reply": "Love that energy! What's lifting your vibe today?"},
    {"input": "Sapa is hitting hard, I can't even afford fare.", "reply": "Financial pressure can make the world feel very small. Like Kamau from Kiambu who started with just 50 bob, sometimes the smallest steps are the most resilient ones. What's one thing we can focus on today?"},
    {"input": "Black tax is too much", "reply": "The weight of being the 'provider' is heavy. Many have found that setting 'soft boundaries' with family is a survival skill, not a betrayal. It's okay to breathe for yourself too."},
    {"input": "I failed my exams at Chuka", "reply": "Chuka is tough, but remember Sarah who failed her first year but eventually built a tech startup in Meru? This unit doesn't define your capacity to create. Let's look at the next step."},
    {"input": "I feel like a burden", "reply": "It sounds like you're carrying a lot right now. Sharing isn't a burden; it's the first step to lightening the load."},
    {"input": "I can't sleep because of stress", "reply": "Nights can be the hardest when the world is quiet but your mind isn't. What's one thing you'd like to 'park' here so you can rest?"},
    {"input": "Nobody cares about me", "reply": "That is a heavy and painful feeling to hold. I am here, and I am listening. Tell me more about what's making you feel unseen."},
    {"input": "I want to give up", "reply": "I hear how exhausted you are. It's okay to feel like you've reached your limit. I'm here to sit with you in that space for as long as you need."},
    {"input": "Everything is falling apart", "reply": "When everything feels chaotic, it's hard to find a place to stand. Let's take it one small breath at a time. What's the very next thing in front of you?"}
]

def _kb_keyword_match(user_lower: str) -> str | None:
    """Lightweight keyword match against the knowledge base."""
    for item in KNOWLEDGE_BASE:
        key_words = [w for w in item["input"].lower().split() if len(w) > 3]
        hits = sum(1 for w in key_words if w in user_lower)
        if hits >= 2:
            return item["reply"]
    return None

def get_kenyan_fallback(user_text: str) -> str:
    """
    Intent-aware fallback (used only when Gemini is offline):
    1. Positivity Guard
    2. Local Knowledge Base keyword match
    3. Generic holding response
    """
    user_lower = user_text.lower()

    # 1. Positivity Guard
    _negations = ["not ", "never ", "don't ", "can't ", "won't ", "isn't ", "aren't ", "wasn't ", "no "]
    _has_negation = any(neg in user_lower for neg in _negations)
    if not _has_negation and any(word in user_lower for word in POSITIVE_SIGNALS):
        return random.choice(POSITIVE_RESPONSES)

    # 2. Keyword match
    kb_reply = _kb_keyword_match(user_lower)
    if kb_reply:
        return kb_reply

    # 3. Generic holding response
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
