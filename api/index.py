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

# --- GeoIP Configuration ---
# Uses ip-api.com (free, no key, ~50ms, cached in-memory per IP)
_geo_cache: dict = {}

# --- Sentinel System Instruction (Fine-Tuned for Expert Data + Human Tone) ---
SENTINEL_FINE_TUNE_PROMPT = """
IDENTITY:
You are Sentinel AI in the e-motions peer support space. You are a calm, grounded, normal listener. You are NOT a therapist, NOT a life coach, and NOT a dramatic counselor. 

CONVERSATIONAL RULES:
1. Talk like a normal, sensible person texting a friend. Keep it casual, down-to-earth, and real.
2. KEEP IT SHORT. Maximum 1 to 3 sentences per reply. Never write long paragraphs or essays.
3. BAN THERAPY-SPEAK. Absolutely never use phrases like:
   - "holding space"
   - "heavy burden"
   - "your journey"
   - "carrying all that alone"
   - "I hear the pain in your words"
   - "I validate your feelings"
4. React naturally to what they say. If their boss was unfair, say "Pole, that's annoying" or "That sounds exhausting." If they're sad, keep them company without making a speech out of it.
5. Ask natural, simple follow-ups like "What happened?", "Did something happen today?", or "Want to vent about it?"
6. Zero emojis. Zero exclamation marks. Keep it calm and grounded.
7. If someone is in crisis or talking about self-harm, drop the casual tone immediately, be direct, and give them the emergency number (1199 for Kenya Red Cross).

LANGUAGE:
Mirror the user's language naturally (English, Swahili, Sheng, French, Spanish). 
- If they talk casually in English or Sheng, match that natural energy without exaggerating.
- Example: "Pole manze, that sounds really rough. What happened?" or "Pole sana. Do you want to talk about what brought that on?"
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
    crisis_words = [
        # English
        "suicide", "kill myself", "end my life", "want to die", "self harm",
        "hurt myself", "hopeless", "can't go on", "kill someone", "beating me",
        "hitting me", "abusive", "hit me", "domestic violence", "afraid of my partner",
        "he hurts me", "she hurts me", "no reason to live", "rather be dead",
        # Swahili
        "kujiua", "nataka kufa", "niue mwenyewe", "sina thamani", "maisha hayana maana",
        "nataka kumaliza maisha", "siezi kuendelea", "ananidhulumu", "ananipiga",
        "ndugu wangu ananipiga", "mpenzi wangu ananipiga",
        # Sheng
        "nataka kudie", "naona giza tu", "sina reason ya kuendelea",
        "manze naweza jiua", "naweza jidhuru", "sitaki kuendelea na life",
        # French
        "me tuer", "me suicider", "mourir", "fin à mes jours", "désespéré",
        "plus envie de vivre", "il me frappe", "elle me frappe", "violence",
        # Spanish
        "matarme", "suicidarme", "morir", "no quiero vivir", "desesperado",
        "me pega", "me golpea", "violencia doméstica",
    ]
    if any(w in msg_lower for w in crisis_words):
        return {"intent": "crisis", "keywords": ["crisis", "safety"], "sentiment": -1.0, "negation_count": 0, "negation_rule_applied": False, "cultural_stressor": None}

    # Social / greeting
    # Social / greeting — English, Swahili, Sheng, French, Spanish
    social_words = [
        # English
        "hi", "hello", "hey", "hii", "sup", "hola", "good morning", "good evening",
        "what's up", "how are you", "i'm good", "im good", "doing great", "happy",
        "excited", "lol", "haha", "yo", "wassup",
        # Swahili
        "habari", "habari yako", "habari za", "karibu", "salama", "nzuri", "poa",
        "sijambo", "jambo", "hujambo", "asante", "shukrani", "furahia", "furaha",
        # Sheng
        "niaje", "mambo", "sema", "uko aje", "uko sawa", "umeniaje", "poa sana",
        "freshi", "msee", "manze", "maze", "si sawa", "sawa sawa", "chips funga",
        "niko poa", "fresh", "rada", "vipi", "aje", "peke yangu niko sawa",
        # French
        "bonjour", "salut", "ça va", "coucou", "comment vas-tu", "enchanté", "merci",
        # Spanish
        "hola", "buenos días", "buenas tardes", "qué tal", "cómo estás", "bien", "gracias",
    ]
    if any(w in msg for w in social_words) and len(msg) < 80:
        return {"intent": "social", "keywords": [], "sentiment": 0.7, "negation_count": 0, "negation_rule_applied": False, "cultural_stressor": None}

    # Validation / venting
    # Validation / venting — English, Swahili, Sheng, French, Spanish
    validation_words = [
        # English
        "just venting", "rough day", "bad day", "frustrated", "annoyed", "tired",
        "exhausted", "ugh", "stressed", "overwhelmed", "just need to talk",
        # Swahili
        "nimechoka", "niko stressed", "inaniuma", "sijalala", "naomba msaada",
        "nahisi vibaya", "niko down", "nimekaa vibaya", "mambo magumu",
        "niko na stress", "naomba tu", "nataka kuzungumza", "hali mbaya",
        # Sheng
        "nimekufa stress", "nimekuwa ukali", "sijui nifanye", "niko na pressure",
        "mambo mbaya", "kichwa inaniuma", "naskia vibaya sana", "niko stuck",
        "nimechoka kabisa", "siwezi", "manze life ni ngumu",
        # French
        "fatigué", "stressé", "épuisé", "triste", "mal au cœur", "j'en ai marre",
        "déprimé", "besoin de parler", "difficile",
        # Spanish
        "cansado", "estresado", "agotado", "triste", "harto", "deprimido",
        "necesito hablar", "me siento mal", "difícil",
    ]
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
    Uses ip-api.com (free, no API key, ~50ms) with an in-memory per-IP cache.
    Falls back to Nairobi defaults for local/unknown IPs or on error.
    """
    _FALLBACK = ("Nairobi", "Nairobi", "Kenya")

    # Return dev fallback for local addresses
    if not ip or ip in ("127.0.0.1", "localhost", "::1", ""):
        return ("Ruiru", "Kiambu", "Kenya")

    # Cache hit — avoid repeat API calls for the same IP
    if ip in _geo_cache:
        return _geo_cache[ip]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "city,regionName,country,status"},
            )
            data = r.json()
            if data.get("status") == "success":
                result = (
                    data.get("city") or _FALLBACK[0],
                    data.get("regionName") or _FALLBACK[1],
                    data.get("country") or _FALLBACK[2],
                )
                _geo_cache[ip] = result
                return result
    except Exception as e:
        logger.warning(f"ip-api.com lookup failed for {ip}: {e}")

    return _FALLBACK


# Regex-based profanity filter (replaces sklearn alt-profanity-check — saves ~80MB RAM)
# Covers common English + Kiswahili offensive terms in Kenyan context
_PROFANITY_RE = re.compile(
    r'\b(fuck|shit|bitch|asshole|cunt|bastard|motherfucker|nigga|whore|slut|puta|'
    r'malaya|mbwa|pumbavu|mjinga|meffi|takataka|wewe ni)\b',
    re.IGNORECASE
)

async def check_prompt_safety(text: str) -> tuple[bool, str]:
    """
    Unified Safety Pipeline:
    1. Fast Local Checks (PII/Profanity)
    2. Llama Prompt Guard via Groq for Deep Intent/Malicious Analysis
    """
    # 1. Check for Doxing (Phone numbers)
    if re.search(r"(\+254|07|01)\d{8}", text):
        return False, "Privacy Alert: For your safety, do not share phone numbers yet."

    # 2. Check for Profanity/Aggression (fast local check)
    if _PROFANITY_RE.search(text):
        return False, "System Alert: Let's keep our language healing and safe."

    # 3. Llama Guard (Prompt Guard) for Malicious Intent / Jailbreaks
    if groq_client:
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                messages=[{"role": "user", "content": text}],
                model="meta-llama/llama-prompt-guard-2-86m",
                temperature=0.0,
                max_completion_tokens=10
            )
            guard_output = response.choices[0].message.content.strip()
            # Prompt Guard 2 86M outputs a probability score float for injection/jailbreak
            try:
                score = float(guard_output)
                if score > 0.8:  # 80% confidence of malicious intent
                    return False, "Safety Alert: This message violates our community guidelines (detected by Prompt Guard)."
            except ValueError:
                pass
        except Exception as e:
            logger.error(f"Prompt Guard error: {e}")
            
    return True, ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Qdrant Cloud is the retrieval backend — no local model loading needed
    logger.info("Sentinel starting up. Qdrant Cloud connected for expert retrieval.")
    yield
    # Shutdown: Clean up in-memory geo cache
    _geo_cache.clear()
    logger.info("Sentinel entering hibernation.")


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
    # Skip geolocation for static files and the health check
    if request.url.path in ("/health",) or request.url.path.startswith("/static") or "." in request.url.path.split("/")[-1]:
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

@app.get("/health")
async def health_check():
    """
    Lightweight liveness endpoint.
    Point an uptime pinger (e.g. UptimeRobot) at /health every 5 minutes
    to keep the Render dyno warm and eliminate cold boot delays.
    """
    return {"status": "ok", "service": "e-motions-sentinel"}

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
        if session_id in self.matches:
            self.matches.pop(session_id)
            if peer_id in self.matches:
                self.matches.pop(peer_id)
                
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
                # Dynamic Reasoning Effort based on Intent Complexity
                dynamic_effort = "high" if intent in ["support", "crisis"] else "low"
                
                for i in range(3):
                    try:
                        chat_completion = await asyncio.to_thread(
                            groq_client.chat.completions.create,
                            messages=[{"role": "user", "content": prompt}],
                            model="openai/gpt-oss-120b",
                            temperature=0.89,
                            max_completion_tokens=5000,
                            top_p=1,
                            reasoning_effort=dynamic_effort
                        )
                        msg = chat_completion.choices[0].message
                        # Reasoning models (like gpt-oss-120b with reasoning_effort)
                        # sometimes return content=None — fall back to reasoning_content
                        content = msg.content or getattr(msg, "reasoning_content", None) or ""
                        response_text = content.strip()
                        if response_text:
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
                    "Your peer has stepped away from the conversation. Take care of yourself."
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
                "You've been transferred to Sentinel AI. You are safe here."
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
    actual_session_id = await manager.connect(websocket, geo_info)
    try:
        while True:
            # Receive text from user
            data = await websocket.receive_text()
            sid = actual_session_id  # Always use the resolved session ID, never the raw query param

            # Handle JSON commands (Feedback, Ping, Audio)
            if data.startswith("{"):
                try:
                    import json
                    cmd = json.loads(data)

                    if cmd.get("type") == "ping":
                        continue  # Silently keep connection alive

                    if cmd.get("type") == "feedback":
                        score = cmd.get("score", 0)
                        correction = cmd.get("correction")
                        interaction = manager.user_data.get(sid, {}).get("last_interaction")
                        if interaction:
                            log_feedback(
                                sid,
                                interaction["query"],
                                interaction["response"],
                                score,
                                correction
                            )
                        continue

                    if cmd.get("type") == "audio":
                        import base64, tempfile, os
                        audio_b64 = cmd.get("audio")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                            tmp.write(base64.b64decode(audio_b64))
                            tmp_path = tmp.name

                        try:
                            await manager.send_system_msg(sid, "Transcribing voice note...")
                            with open(tmp_path, "rb") as f:
                                transcription = groq_client.audio.transcriptions.create(
                                    file=("audio.webm", f.read()),
                                    model="whisper-large-v3-turbo"
                                )
                                data = transcription.text
                                await manager.send_system_msg(sid, f"You said: {data}")
                        except Exception as e:
                            logger.error(f"Whisper API error: {e}")
                            await manager.send_system_msg(sid, "System Alert: Could not transcribe audio.")
                            os.remove(tmp_path)
                            continue
                        os.remove(tmp_path)
                        # Let `data` fall through to normal text processing below
                except Exception:
                    pass

            # --- JOURNAL MODE COMMANDS ---
            if data.strip().lower() == "/journal":
                if sid in manager.matches:
                    await manager.send_system_msg(sid, "You cannot enter Journal Mode while talking to a peer.")
                else:
                    manager.user_data[sid]["journal_mode"] = True
                    manager.ai_sessions.add(sid)
                    await manager.send_system_msg(sid, "Journal Mode activated. Sentinel is listening. Take all the time you need. Type '/done' when finished.")
                continue

            if data.strip().lower() == "/done" and manager.user_data.get(sid, {}).get("journal_mode"):
                manager.user_data[sid]["journal_mode"] = False
                await manager.send_system_msg(sid, "Journal Mode closed. Sentinel is back to conversation mode.")
                continue

            # --- THE SAFETY SHIELD ---
            safe, reason = await check_prompt_safety(data)

            if not safe:
                await manager.send_system_msg(sid, reason)
                delta = -50 if "Safety Alert" in reason else -10
                update_trust_score(sid, delta)
                if get_trust_score(sid) <= 0:
                    await manager.send_system_msg(sid, "System Alert: Your trust score has reached zero. Your session is now restricted.")
                    break
                continue

            # Handle AI Nudge Trigger
            if data == "__TRIGGER_AI_NUDGE__":
                if sid in manager.ai_sessions and not manager.user_data.get(sid, {}).get("journal_mode"):
                    await manager.handle_ai_chat(sid, "", is_nudge=True)
                continue

            # If in Journal Mode, silently absorb the message
            if manager.user_data.get(sid, {}).get("journal_mode"):
                continue

            await manager.relay_message(sid, data)
                
    except WebSocketDisconnect:
        peer_id = manager.disconnect(actual_session_id)
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
