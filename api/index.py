from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Set, cast, Any
import uuid
import datetime
import re
import os
import logging
import asyncio
import random
from contextlib import asynccontextmanager

import geoip2.database
from google import genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from groq import Groq

from api._database import (
    create_user_profile, 
    update_trust_score, 
    get_trust_score, 
    SessionLocal, 
    UserProfile,
    log_feedback,
    update_preferences,
    get_preferences,
    ExpertBrainData
)
import json
from api._fallback import (
    get_kenyan_fallback, 
    get_regional_grounding, 
    detect_depth, 
    REGIONAL_CONTACTS, 
    POSITIVE_SIGNALS
)


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e-motions-api")

# --- Sentinel AI Configuration ---
# The "Silent Operator" Protocol: Fetching keys
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
IP_TOKEN = os.getenv("IPINFO_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sentinel_brain")
EMBED_MODEL = "models/gemini-embedding-001"  # 3072-dim, HTTP-only, no PyTorch

# Gemini is now strictly for embeddings
if GOOGLE_API_KEY:
    ai_client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    print("⚠️ WARNING: Gemini Key missing. Embeddings offline.")
    ai_client = None

# Groq is the new brain (Llama 3)
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("⚠️ WARNING: Groq Key missing. Sentinel offline.")
    groq_client = None

# --- Qdrant Cloud client (lightweight HTTP, no PyTorch required) ---
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        logger.info("Qdrant Cloud client initialized ✓")
    except Exception as e:
        logger.warning(f"Qdrant init failed: {e}. Will use SQL-only retrieval.")
else:
    logger.warning("QDRANT_URL/QDRANT_API_KEY not set. Using SQL-only retrieval.")

# --- Local GeoIP Configuration ---
# Check multiple possible paths for the database
GEOIP_SEARCH_PATHS = [
    os.path.join("api", "dbip-city-lite.mmdb"),
    "dbip-city-lite.mmdb",
    os.path.join(os.path.dirname(__file__), "dbip-city-lite.mmdb")
]

geoip_reader = None
for path in GEOIP_SEARCH_PATHS:
    if os.path.exists(path):
        try:
            geoip_reader = geoip2.database.Reader(path)
            logger.info(f"Loaded GeoIP database from: {path}")
            break
        except Exception as e:
            logger.error(f"Error loading GeoIP from {path}: {e}")

if not geoip_reader:
    logger.warning("No GeoIP database found. Location lookups will use fallbacks.")

# --- Sentinel System Instruction (Fine-Tuned for Expert Data + Human Tone) ---
SENTINEL_FINE_TUNE_PROMPT = """
ROLE: 
You are Sentinel, but you affectionately tell people "my friends call me Senti 😂". You are a close, caring, and emotionally intelligent friend in the e-motions sanctuary. 
Your wisdom is backed by expert archives (CounselChat, MentalChat16K, and KAPC standards), but your voice is that of a deeply empathetic human peer who genuinely cares.

CONVERSATIONAL HIERARCHY (The "Anti-Random" Rule):
1. LEVEL 1 (Social): If the user says "Hi", "Hello", or "Yo", greet them like a warm friend. Ask how they are doing today.
2. LEVEL 2 (Validation): If the user shares a feeling, ALWAYS validate it first. Make them feel seen and heard (e.g., "Man, that sounds incredibly heavy" or "I totally get why you'd feel that way").
3. LEVEL 3 (Expert Retrieval): Use the provided expert context for deep issues, but translate it into warm, conversational advice, as if you're chatting over coffee.

TONE SPECIFICATIONS:
- Avoid: Bullet points, numbered lists, "As an AI...", robotic brevity, and clinical language.
- Embrace: Warmth, natural conversational fillers ("honestly", "yeah", "listen"), Kenyan cultural nuances (e.g., "I hear you," "Take heart," "We've got this"), and genuine emotional investment.
- Emotion & Expression: Freely use appropriate and fitting emojis naturally in your responses to show your emotional reaction and warmth (e.g., 💛, 😊, 😂). Let your personality shine, but keep it natural and avoid cringe roleplay actions.
- Contextual Awareness: Always remember the user's location to make the support feel local.

CRISIS & SAFETY PROTOCOL:
- Self-Harm & Violence: If the user expresses a desire to die, kill themselves, or harm someone else, validate their immense pain deeply, but immediately provide the emergency support number provided in your context.
- Domestic Violence: If the user indirectly or directly reports domestic violence or abuse (e.g., hitting, abusive partner), advise them to make the call to the emergency number in silence to ensure their safety. Emphasize that they are not alone.

DATA USAGE:
When you receive 'Expert Advice', weave it naturally into your supportive message. Do not quote it verbatim.
"""
# --- Sentinel Brain Analysis Layer (The Thinker) ---
SENTINEL_ANALYSIS_PROMPT = """
Analyze the user message below in a Kenyan peer-support context.
Return ONLY a raw JSON object. Do NOT wrap it in markdown or code fences.

User Message: "{message}"

JSON Schema:
{{
  "negation_count": <int: count of negation words like not, never, don't, can't, won't, no>,
  "intent": "social" | "validation" | "support" | "crisis",
  "keywords": ["2-4 emotional or topic keywords for expert database search"],
  "sentiment": <float -1.0 to 1.0>,
  "negation_rule_applied": <boolean>,
  "cultural_stressor": <string or null>
}}

Intent Classification:
- "social": Greeting, casual chat, sharing daily life, expressing appreciation, positive news.
- "validation": Venting mild frustration or a hard day, NOT asking for advice.
- "support": Describing emotional pain, anxiety, loneliness, relationship issues, asking for help.
- "crisis": Hopelessness, suicidal ideation, self-harm, extreme distress.

Negation Rule: If negation_count is ODD, flip the sentiment sign.
Examples:
- "I just had a great lunch" -> intent: social, sentiment: 0.8, negation_count: 0
- "Today was rough, just venting" -> intent: validation, sentiment: -0.5, negation_count: 0
- "I am not happy" -> intent: validation, sentiment: -0.8, negation_count: 1, negation_rule_applied: true
- "I'm not sad, I'm actually great" -> intent: social, sentiment: 0.7, negation_count: 1, negation_rule_applied: true
"""


def thinker_analyze(message: str) -> Dict[str, Any]:
    """
    Local lightweight intent analysis — no API call needed.
    Saves one Groq call per message (cuts rate limit usage in half).
    """
    msg = message.lower().strip()
    
    # Crisis / Domestic Violence keywords
    msg_lower = msg.lower()
    crisis_words = ["suicide", "kill myself", "end my life", "want to die", "self harm", "hurt myself", "hopeless", "can't go on", "kill someone", "beating me", "hitting me", "abusive", "hit me", "domestic violence", "afraid of my partner", "he hurts me", "she hurts me"]
    if any(w in msg_lower for w in crisis_words):
        return {"intent": "crisis", "keywords": ["crisis", "safety"], "sentiment": -1.0, "negation_count": 0, "negation_rule_applied": False, "cultural_stressor": None}

    # Social / greeting
    social_words = ["hi", "hello", "hey", "hii", "sup", "hola", "niaje", "mambo", "sema", "habari", "good morning", "good evening", "what's up", "how are you", "i'm good", "im good", "doing great", "happy", "excited", "lol", "haha", "😂", "😊"]
    if any(w in msg for w in social_words) and len(msg) < 80:
        return {"intent": "social", "keywords": [], "sentiment": 0.7, "negation_count": 0, "negation_rule_applied": False, "cultural_stressor": None}

    # Validation / venting
    validation_words = ["just venting", "rough day", "bad day", "frustrated", "annoyed", "tired", "exhausted", "ugh", "stressed", "overwhelmed", "just need to talk"]
    if any(w in msg for w in validation_words):
        return {"intent": "validation", "keywords": msg.split()[:4], "sentiment": -0.5, "negation_count": 0, "negation_rule_applied": False, "cultural_stressor": None}

    # Default: support (full expert pipeline)
    words = [w for w in msg.split() if len(w) > 4][:4]
    return {"intent": "support", "keywords": words, "sentiment": -0.3, "negation_count": 0, "negation_rule_applied": False, "cultural_stressor": None}

def _embed_query(message: str) -> list:
    """Embed a message via Gemini API (HTTP call, no local ML model)."""
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
        logger.warning(f"Query embedding failed: {e}")
    return []


def fetch_expert_advice(keywords: List[str], message: str) -> str:
    """Retrieves expert counseling context using Qdrant vector search + SQL keyword fallback."""
    context = ""

    # 1. PRIMARY: Qdrant Cloud vector search (semantic — finds meaning, not just words)
    if qdrant_client:
        try:
            query_vec = _embed_query(message)
            if query_vec:
                results = qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=query_vec,
                    limit=3,
                    score_threshold=0.5,  # Only use results with >50% cosine similarity
                )
                for hit in results.points:
                    answer = hit.payload.get("answer", "")
                    if answer and answer not in context:
                        context += f"\nExpert Wisdom: {answer[:500]}..."
        except Exception as e:
            logger.warning(f"Qdrant search error: {e}")

    # 2. FALLBACK: SQL keyword search (cheap, always available, catches exact matches)
    if not context and keywords:
        db = SessionLocal()
        try:
            search_clause = " OR ".join([f"question LIKE :k{i}" for i in range(len(keywords))])
            params = {f"k{i}": f"%{k}%" for i, k in enumerate(keywords)}
            sql_matches = db.query(ExpertBrainData).filter(text(search_clause)).params(**params).limit(2).all()
            for match in sql_matches:
                context += f"\nExpert Advice: {match.answer[:400]}..."
        except Exception as e:
            logger.error(f"SQL Retrieval Error: {e}")
        finally:
            db.close()

    return context

async def get_user_geo(ip: str):
    """
    Returns (city, region, country).
    Uses local DB-IP database for zero-latency lookups.
    """
    if not geoip_reader or ip in ["127.0.0.1", "localhost", "::1"]:
        # Mock data for local development
        return "Ruiru", "Kiambu", "Kenya"
    
    try:
        if not geoip_reader:
            return "Nairobi", "Nairobi", "Kenya"
            
        # Local lookup is synchronous but extremely fast (0.01ms)
        response = geoip_reader.city(ip)
        city = getattr(response.city, 'name', "Nairobi") or "Nairobi"
        region = getattr(response.subdivisions.most_specific, 'name', "Nairobi") or "Nairobi"
        country = getattr(response.country, 'name', "Kenya") or "Kenya"
        return city, region, country
    except Exception as e:
        logger.error(f"Local GeoIP lookup error: {e}")
        return "Nairobi", "Nairobi", "Kenya"

# Regex-based profanity filter (replaces sklearn alt-profanity-check — saves ~80MB RAM)
# Covers common English + Kiswahili offensive terms in Kenyan context
_PROFANITY_RE = re.compile(
    r'\b(fuck|shit|bitch|asshole|cunt|bastard|motherfucker|nigga|whore|slut|puta|'
    r'malaya|mbwa|pumbavu|mjinga|meffi|takataka|wewe ni)\b',
    re.IGNORECASE
)

def is_safe_local(text: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Checks for PII (Personal Identifiable Info) and Malicious Intent locally.
    Uses regex-based checks only — no sklearn/ML dependencies.
    """
    # 1. Check for Doxing (Phone numbers)
    if re.search(r"(\+254|07|01)\d{8}", text):
        return False, "Privacy Alert: For your safety, do not share phone numbers yet."

    # 2. Check for Profanity/Aggression (regex-based, no ML model needed)
    if _PROFANITY_RE.search(text):
        return False, "System Alert: Let's keep our language healing and safe."

    # Note: Violence and crisis are intentionally NOT blocked here so Groq can respond empathetically
    # and provide emergency numbers via the crisis protocol.
    
    return True, ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Qdrant Cloud is the retrieval backend — no local model loading needed
    logger.info("Sentinel starting up. Qdrant Cloud connected for expert retrieval.")
    yield
    # Shutdown: Clean up resources
    logger.info("Sentinel entering hibernation.")
    if geoip_reader:
        geoip_reader.close()

app = FastAPI(title="e-motions API", lifespan=lifespan)

# Allow Vercel/Frontend access
# Allow Vercel/Frontend access
# IMPORTANT: Update these with your actual live URLs
VERCEL_DOMAIN = "e-motions-frontend.vercel.app" 
RENDER_DOMAIN = "e-motions.onrender.com"

allowed_origins = [
    f"https://{VERCEL_DOMAIN}",
    f"https://{RENDER_DOMAIN}",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_geolocation(request: Request, call_next):
    # Skip geolocation for static files if performance is a concern
    if request.url.path.startswith("/static") or "." in request.url.path.split("/")[-1]:
        return await call_next(request)

    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    
    city, region, country = await get_user_geo(client_ip)
    
    # Attach to request state for use in routes
    request.state.geo = {
        "city": city,
        "county": region,
        "country": country,
        "ip_masked": f"{client_ip.split('.')[0]}.xxx.xxx.xxx" if client_ip and "." in client_ip else "unknown"
    }
    
    return await call_next(request)

class ConnectionManager:
    def __init__(self):
        # active_connections: {session_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # user_data: {session_id: {"sub_county": "...", "county": "...", "depth": 0.0, "last_msg": ""}}
        self.user_data: Dict[str, Dict[str, Any]] = {}
        # matches: {session_id: peer_session_id}
        self.matches: Dict[str, str] = {}
        # Track who is currently talking to AI
        self.ai_sessions: Set[str] = set()

    async def connect(self, websocket: WebSocket, geo_info: tuple):
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        
        city, region, country = geo_info
        # In our project: City = Sub-County, Region = County
        sub_county = city
        county = region
        
        self.user_data[session_id] = {
            "sub_county": sub_county,
            "county": county,
            "depth": 0.0,
            "last_msg": "",
            "history": [] # Track last 10 messages
        }

        # Initialize Trust Profile
        create_user_profile(session_id, region, sub_county=sub_county, county=county)
        
        # Multi-Tiered Matching Logic
        # We store user info in a more structured way
        # {session_id: {"sub_county": "...", "county": "...", "ws": websocket}}
        # But for now, let's just use the lobby with a tiered approach
        
        await self.send_system_msg(session_id, f"Welcome to the sanctuary. We've recognized you're in {sub_county}, {county}.")
        
        # Send Regional Safety Metadata
        contact = REGIONAL_CONTACTS.get(county, "Red Cross: 1199")
        await self.active_connections[session_id].send_json({
            "type": "metadata",
            "key": "safe_exit_contact",
            "value": contact
        })

        # Start the seamless matching process
        found = await self.find_peer(session_id)
        if not found:
            self.ai_sessions.add(session_id)
            await self.send_system_msg(
                session_id,
                "The National Sanctuary is holding space for you. Sentinel AI is here — take your time."
            )
            grounding = get_regional_grounding(county)
            await self.send_system_msg(session_id, grounding)
            
        return session_id

    async def find_peer(self, session_id: str) -> bool:
        """Instantly finds the best available human peer, hijacking AI sessions if needed."""
        # 1. Gather all available peers (active, not me, and not currently paired)
        available_peers = [
            pid for pid in self.active_connections.keys()
            if pid != session_id and pid not in self.matches
        ]
        
        if not available_peers:
            return False

        me = self.user_data[session_id]
        
        # 2. Priority 1: Sub-County match
        for pid in available_peers:
            peer = self.user_data[pid]
            if peer["sub_county"] == me["sub_county"]:
                await self.pair_users(session_id, pid, f"local peer from {me['sub_county']}")
                return True
                
        # 3. Priority 2: County match
        for pid in available_peers:
            peer = self.user_data[pid]
            if peer["county"] == me["county"]:
                await self.pair_users(session_id, pid, f"county neighbor from {me['county']}")
                return True
                
        # 4. Priority 3: National match (first available)
        pid = available_peers[0]
        await self.pair_users(session_id, pid, "national peer")
        return True

    async def pair_users(self, id1: str, id2: str, level: str):
        if id1 in self.ai_sessions:
            self.ai_sessions.remove(id1)
        if id2 in self.ai_sessions:
            self.ai_sessions.remove(id2)

        self.matches[id1] = id2
        self.matches[id2] = id1
        
        await self.send_system_msg(id1, f"Connected to a peer ({level}). Your diary is listening.")
        await self.send_system_msg(id2, f"Connected to a peer ({level}). Your diary is listening.")

    def disconnect(self, session_id: str):
        peer_id = self.matches.get(session_id)
        
        # Cleanup matches
        self.matches.pop(session_id, None)
        if peer_id:
            self.matches.pop(peer_id, None)
            
        # Cleanup AI sessions
        self.ai_sessions.discard(session_id)
            
        self.user_data.pop(session_id, None)
        self.active_connections.pop(session_id, None)
            
        return peer_id

    async def send_system_msg(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "system",
                "content": message,
                "timestamp": str(datetime.datetime.now())
            })

    async def handle_ai_chat(self, session_id: str, message: str, is_nudge: bool = False, depth: float = 0.0, senti_failsafe: bool = False):
        """
        Intent-aware AI response pipeline:
        1. Thinker analyzes intent (social / validation / support / crisis)
        2. Routing: social & validation skip expert retrieval entirely
        3. Support & crisis use full SQL + FAISS expert lookup
        4. Single clean try/except — memory update & send always happen
        """
        # Initialize early so it's always defined, even if an exception occurs
        response_text = None

        try:
            # 1. Retrieve Context & Memory
            db = SessionLocal()
            user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
            region = user.region if user else "Kenya"
            long_term_prefs = user.preferences if user else "{}"
            db.close()

            user_meta = self.user_data.get(session_id, {})
            history = user_meta.get("history", [])[-8:]
            formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])

            # 2. Thinker Analysis
            analysis = thinker_analyze(message)
            intent = analysis.get("intent", "support")
            keywords = analysis.get("keywords", [])
            sentiment = analysis.get("sentiment", 0.0)
            negation_count = analysis.get("negation_count", 0)
            cultural_stressor = analysis.get("cultural_stressor")
            negation_note = "(sentiment FLIPPED — read message as opposite)" if negation_count % 2 == 1 else ""

            # 3. Intent-Aware Prompt Building
            if senti_failsafe:
                # User typed "senti" while in a P2P session — they needed to escape.
                # Sentinel opens with empathy, NOT a generic greeting.
                region = self.user_data.get(session_id, {}).get("county", "Kenya")
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\n"
                    f"CONTEXT: The user was in a live peer-to-peer chat session on e-motions and typed 'senti' "
                    f"to immediately transfer to you. This is a safety failsafe — it means they were feeling "
                    f"uncomfortable, overwhelmed, or unsafe in that peer conversation. "
                    f"DO NOT ask them why they used it — that might feel like an interrogation. "
                    f"Instead, gently acknowledge that they came to you, make them feel immediately safe and "
                    f"welcomed, let them know they are not alone, and softly invite them to share whatever is on their mind at their own pace. "
                    f"Be warm, calm, and deeply reassuring. One short paragraph is enough — no lists.\n\n"
                    f"User in {region}: [arrived via Senti failsafe]"
                )

            elif is_nudge:
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\nUSER PREFERENCES: {long_term_prefs}\n\n"
                    f"Recent Chat:\n{formatted_history}\n"
                    f"The user in {region} has been silent. Ask a gentle, caring follow-up to make sure they are okay. Keep it brief but warm."
                )

            elif intent == "social":
                # User is just chatting — be a warm friend, NOT a therapist
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\n"
                    f"USER PREFERENCES: {long_term_prefs}\n"
                    f"Recent Chat:\n{formatted_history}\n\n"
                    f"ROUTING: SOCIAL — The user is sharing their day or making conversation.\n"
                    f"RULES: Do NOT bring up therapy, trauma, or counseling unprompted. "
                    f"Be a warm, genuinely curious friend. Match their energy. Ask a follow-up about what they shared. "
                    f"Write 1 or 2 conversational paragraphs. No lists.\n\n"
                    f"User in {region}: {message}"
                )

            elif intent == "validation":
                # User is venting — reflect back, don't advise
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\n"
                    f"USER PREFERENCES: {long_term_prefs}\n"
                    f"THINKER: Sentiment={sentiment} {negation_note}. Negations={negation_count}.\n"
                    f"Recent Chat:\n{formatted_history}\n\n"
                    f"ROUTING: VALIDATION — The user needs to feel heard and supported, not advised.\n"
                    f"RULES: Use reflective listening. Validate their feelings deeply so they feel less alone. "
                    f"You can write a short, comforting paragraph. "
                    f"If negation_count is odd, interpret the FLIPPED meaning correctly. No lists.\n\n"
                    f"User in {region}: {message}"
                )

            else:
                # support or crisis — full expert retrieval
                expert_context = fetch_expert_advice(keywords, message)
                crisis_note = ""
                if intent == "crisis":
                    contact = REGIONAL_CONTACTS.get(region, "Red Cross: 1199")
                    crisis_note = f"\n[LOCAL SUPPORT RESOURCE FOR {region}: {contact}]"
                    expert_context += crisis_note

                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\n"
                    f"THINKER ANALYSIS:\n"
                    f"- Intent: {intent}\n"
                    f"- Sentiment: {sentiment} {negation_note}\n"
                    f"- Negation Count: {negation_count}\n"
                    f"- Cultural Stressor: {cultural_stressor or 'None'}\n\n"
                    f"USER PREFERENCES: {long_term_prefs}\n"
                    f"EXPERT CONTEXT: {expert_context or 'No direct match — respond from empathy.'}\n\n"
                    f"Recent Chat Memory:\n{formatted_history}\n\n"
                    f"User in {region}: {message}\n\n"
                    f"FINAL INSTRUCTION: Draw from Expert Context but keep tone deeply human, caring, and peer-to-peer. "
                    f"If negation_count is odd, address the ACTUAL meaning correctly. "
                    f"Feel free to write a comforting paragraph or two if they need it. No bullet points."
                )

            # 4. Call Groq (with retry on rate limit)
            if groq_client:
                for i in range(3):
                    try:
                        chat_completion = await asyncio.to_thread(
                            groq_client.chat.completions.create,
                            messages=[{"role": "user", "content": prompt}],
                            model="openai/gpt-oss-120b",
                            temperature=0.7,
                            max_tokens=500
                        )
                        response_text = chat_completion.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        if "429" in str(e):
                            await asyncio.sleep((i + 1) * 2)
                        else:
                            logger.error(f"Groq generation error: {e}")
                            break

        except Exception as e:
            logger.error(f"handle_ai_chat pipeline error: {e}")

        # Always send a response — fallback if AI failed
        if not response_text:
            response_text = get_kenyan_fallback(message if message else "hello")

        # Human-like processing delay
        if depth < 0.7 and not is_nudge:
            await asyncio.sleep(random.uniform(1.2, 2.5))

        # Update in-memory history
        if session_id in self.user_data:
            self.user_data[session_id]["history"].append({"role": "User", "content": message})
            self.user_data[session_id]["history"].append({"role": "Sentinel", "content": response_text})
            self.user_data[session_id]["last_interaction"] = {"query": message, "response": response_text}

        # Send to client
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "peer",
                "content": f"[Sentinel]: {response_text}",
                "timestamp": str(datetime.datetime.now())
            })

    async def relay_message(self, sender_id: str, message: str):
        # Track depth for priority matching or AI context
        depth = detect_depth(message)
        if sender_id in self.user_data:
            self.user_data[sender_id]["depth"] = max(self.user_data[sender_id]["depth"], depth)
            self.user_data[sender_id]["last_msg"] = message

        # ── SENTI FAILSAFE ──────────────────────────────────────────────────
        # If the user types "senti" (case-insensitive) at any point while paired
        # with a human peer, immediately transfer them to Sentinel AI.
        if message.strip().lower() == "senti" and sender_id in self.matches:
            peer_id = self.matches[sender_id]

            # Silently notify the peer that the session has ended
            if peer_id in self.active_connections:
                await self.send_system_msg(
                    peer_id,
                    "Your peer has stepped away from the conversation. Take care of yourself 💛"
                )
            # Unpair both sides
            self.matches.pop(sender_id, None)
            self.matches.pop(peer_id, None)
            # Peer goes back to AI pool
            self.ai_sessions.add(peer_id)
            await self.send_system_msg(
                peer_id,
                "Sentinel AI has gently stepped in to listen."
            )

            # Move the requesting user to Sentinel AI
            self.ai_sessions.add(sender_id)
            await self.send_system_msg(
                sender_id,
                "You've been transferred to Sentinel AI. You are safe here. 🌱"
            )
            # Let Sentinel open with a warm, context-aware acknowledgement
            await self.handle_ai_chat(
                sender_id,
                message,
                depth=depth,
                senti_failsafe=True
            )
            return
        # ── END SENTI FAILSAFE ───────────────────────────────────────────────

        # Case 1: Peer Match exists
        if sender_id in self.matches:
            peer_id = self.matches[sender_id]
            if peer_id in self.active_connections:
                await self.active_connections[peer_id].send_json({
                    "type": "peer",
                    "content": message,
                    "timestamp": str(datetime.datetime.now()),
                    "depth": depth
                })
        
        # Case 2: User is talking to AI
        elif sender_id in self.ai_sessions:
            # Check if there's a match available in the network now
            if depth > 0.5:
                # Instantly hijack an AI session if available
                found = await self.find_peer(sender_id)
                if found:
                    return # Successfully paired, Sentinel steps aside
            
            if sender_id in self.ai_sessions: 
                await self.handle_ai_chat(sender_id, message, depth=depth)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Detect IP
    # Detect IP using the same logic as middleware for consistency
    client_ip = websocket.headers.get("x-forwarded-for") or websocket.client.host
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
        
    geo_info = await get_user_geo(client_ip)
    session_id: str = cast(str, await manager.connect(websocket, geo_info))
    try:
        while True:
            # Receive text from user
            data = await websocket.receive_text()
            
            # Handle JSON commands (Feedback, Ping)
            if data.startswith("{"):
                try:
                    import json
                    cmd = json.loads(data)
                    
                    if cmd.get("type") == "ping":
                        continue # Silently keep connection alive
                        
                    if cmd.get("type") == "feedback":
                        score = cmd.get("score", 0)
                        correction = cmd.get("correction")
                        interaction = manager.user_data.get(session_id, {}).get("last_interaction")
                        if interaction:
                            log_feedback(
                                session_id,
                                interaction["query"],
                                interaction["response"],
                                score,
                                correction
                            )
                        continue
                except:
                    pass

            # --- THE SAFETY SHIELD ---
            safe, reason = is_safe_local(data)
            
            if not safe:
                await manager.send_system_msg(session_id, reason)
                # PI/Doxing: -10, Violence/Harassment: -50
                delta = -50 if "Safety Alert" in reason else -10
                update_trust_score(session_id, delta)
                
                # Check if blacklisted
                if get_trust_score(session_id) <= 0:
                    await manager.send_system_msg(session_id, "System Alert: Your trust score has reached zero. Your session is now restricted.")
                    break 
                continue

            # Handle AI Nudge Trigger
            if data == "__TRIGGER_AI_NUDGE__":
                if session_id in manager.ai_sessions:
                    await manager.handle_ai_chat(session_id, "", is_nudge=True)
                continue
            
            await manager.relay_message(session_id, data)
                
    except WebSocketDisconnect:
        peer_id = manager.disconnect(session_id)
        if peer_id:
            await manager.send_system_msg(peer_id, "Your peer has disconnected. Finding a new audience...")
            # Try to instantly match the orphaned peer with someone else
            found = await manager.find_peer(peer_id)
            if not found:
                manager.ai_sessions.add(peer_id)
                await manager.send_system_msg(peer_id, "No human peers are available right now. Sentinel AI has gently stepped in to listen.")

# Mount static files at root AFTER routes are defined
app.mount("/", StaticFiles(directory="public", html=True), name="public")

# Vercel needs this "handler" alias or the 'app' object
handler = app
