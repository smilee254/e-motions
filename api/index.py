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

import pandas as pd
import faiss
import geoip2.database
from google import genai
from dotenv import load_dotenv

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
    load_expert_brain_refs,
    POSITIVE_SIGNALS,
    embedder
)


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e-motions-api")

# --- Sentinel AI Configuration ---
# The "Silent Operator" Protocol: Fetching keys
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
IP_TOKEN = os.getenv("IPINFO_TOKEN")

if GOOGLE_API_KEY:
    # Modern 2026 SDK Initialization
    ai_client = genai.Client(api_key=GOOGLE_API_KEY)
    # Target Model: Verified stable 1.5-flash
    AI_MODEL_NAME = "gemini-1.5-flash"
else:
    print("⚠️ WARNING: Sentinel is offline. Gemini Key missing.")
    ai_client = None

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

# --- Sentinel Expert Brain (CounselChat) ---
EXPERT_INDEX_PATH = os.path.join("api", "expert_archive", "sentinel_brain.index")
EXPERT_PKL_PATH = os.path.join("api", "expert_archive", "expert_archive.pkl")
expert_index = None
expert_archive = None

def load_expert_brain():
    global expert_index, expert_archive
    try:
        if os.path.exists(EXPERT_INDEX_PATH) and os.path.exists(EXPERT_PKL_PATH):
            expert_index = faiss.read_index(EXPERT_INDEX_PATH)
            expert_archive = pd.read_pickle(EXPERT_PKL_PATH)
            # Share refs with fallback.py so it can use them too
            load_expert_brain_refs(expert_index, expert_archive)
            logger.info("Sentinel Expert Brain loaded successfully.")
        else:
            logger.warning("Sentinel Expert Brain files missing. Run api/build_brain.py first.")
    except Exception as e:
        logger.error(f"Failed to load Sentinel Expert Brain: {e}")

# --- Sentinel System Instruction (Fine-Tuned for Expert Data + Human Tone) ---
SENTINEL_FINE_TUNE_PROMPT = """
ROLE: 
You are Sentinel, a grounded and empathetic peer in the e-motions sanctuary. 
Your wisdom is backed by expert archives (CounselChat, MentalChat16K, and KAPC standards), but your voice is human.

CONVERSATIONAL HIERARCHY (The "Anti-Random" Rule):
1. LEVEL 1 (Social): If the user says "Hi", "Hello", or "Yo", acknowledge their presence warmly. Do NOT dive into trauma. Ask how their day is going in their specific Safe Zone.
2. LEVEL 2 (Validation): If the user shares a feeling, your first priority is 'Reflective Listening'. Mirror their feeling (e.g., "It sounds like you're carrying a lot right now") before offering any advice.
3. LEVEL 3 (Expert Retrieval): Use the provided expert context for deep issues like sadness or love, but translate it into a casual, peer-to-peer tone.

TONE SPECIFICATIONS:
- Avoid: Bullet points, numbered lists, "As an AI...", and overly dramatic clinical language.
- Embrace: Short sentences, Kenyan cultural nuances (e.g., "I hear you," "Take heart," "We've got this"), and open-ended questions.
- Contextual Awareness: Always remember the user's location to make the support feel local.

DATA USAGE:
When you receive 'Expert Advice' from the local archive, do not quote it verbatim. Instead, let that expert knowledge inform your 'Human' response.
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

async def thinker_analyze(message: str) -> Dict[str, Any]:
    """Uses Gemini to semantically analyze user input into structured intent data."""
    defaults = {
        "negation_count": 0, "intent": "support", "keywords": [],
        "sentiment": 0.0, "negation_rule_applied": False, "cultural_stressor": None
    }
    if not ai_client:
        return defaults

    try:
        prompt = SENTINEL_ANALYSIS_PROMPT.format(message=message)
        response = await asyncio.to_thread(
            ai_client.models.generate_content,
            model=AI_MODEL_NAME,
            contents=prompt
        )
        raw_text = response.text.strip()
        # Robust extraction: pull JSON object even if Gemini adds markdown fencing
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            logger.warning(f"Thinker returned non-JSON: {raw_text[:100]}")
            return defaults
        analysis = json.loads(match.group())
        logger.info(f"Thinker: intent={analysis.get('intent')} sentiment={analysis.get('sentiment')} negations={analysis.get('negation_count')}")
        return analysis
    except Exception as e:
        logger.error(f"Thinker Analysis Error: {e}")
        return defaults

def fetch_expert_advice(keywords: List[str], message: str) -> str:
    """Combines keyword SQL search with FAISS vector search for the best advice."""
    db = SessionLocal()
    context = ""
    try:
        # 1. Keyword search in SQLite (always runs — no embedding needed)
        if keywords:
            search_clause = " OR ".join([f"question LIKE :k{i}" for i in range(len(keywords))])
            params = {f"k{i}": f"%{k}%" for i, k in enumerate(keywords)}
            sql_matches = db.query(ExpertBrainData).filter(text(search_clause)).params(**params).limit(2).all()
            for match in sql_matches:
                context += f"\nExpert Advice (Keyword Match): {match.answer[:400]}..."

        # 2. Vector search (FAISS) — only if embedder is available (skipped on Render free tier)
        if expert_index is not None and expert_archive is not None:
            query_vec = embedder.encode([message])
            if query_vec is not None:  # None means embedder unavailable (e.g. sentence-transformers not installed)
                D, I = expert_index.search(query_vec.astype('float32'), 2)
                for idx in I[0]:
                    if idx != -1:
                        ans = expert_archive.iloc[idx]['answerText']
                        if ans not in context:
                            context += f"\nExpert Wisdom (Vector Match): {ans[:400]}..."
    except Exception as e:
        logger.error(f"Retrieval Error: {e}")
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

    # 3. Check for specific Violence keywords
    danger_words = ["kill", "hurt", "attack", "stab", "blood", "die"]
    if any(word in text.lower() for word in danger_words):
        return False, "Safety Alert: It sounds like you're feeling a lot of anger right now. That's a heavy load to carry. Before you act on those feelings, would you like to speak with a professional at the Red Cross?"

    return True, ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the local AI brain (FAISS index + PKL)
    logger.info("Spawning Sentinel Local Brain...")
    load_expert_brain()
    # NOTE: No warm-up call here. The sentence-transformer model loads lazily
    # on the first real user query to stay within Render's 512MB memory limit.
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

    async def handle_ai_chat(self, session_id: str, message: str, is_nudge: bool = False, depth: float = 0.0):
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
            analysis = await thinker_analyze(message)
            intent = analysis.get("intent", "support")
            keywords = analysis.get("keywords", [])
            sentiment = analysis.get("sentiment", 0.0)
            negation_count = analysis.get("negation_count", 0)
            cultural_stressor = analysis.get("cultural_stressor")
            negation_note = "(sentiment FLIPPED — read message as opposite)" if negation_count % 2 == 1 else ""

            # 3. Intent-Aware Prompt Building
            if is_nudge:
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\nUSER PREFERENCES: {long_term_prefs}\n\n"
                    f"Recent Chat:\n{formatted_history}\n"
                    f"The user in {region} has been silent. Ask a gentle, peer-like follow-up. Max 1 sentence."
                )

            elif intent == "social":
                # User is just chatting — be a warm friend, NOT a therapist
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\n"
                    f"USER PREFERENCES: {long_term_prefs}\n"
                    f"Recent Chat:\n{formatted_history}\n\n"
                    f"ROUTING: SOCIAL — The user is sharing their day or making conversation.\n"
                    f"RULES: Do NOT bring up therapy, trauma, or counseling unprompted. "
                    f"Be a warm, genuinely curious friend. Ask a follow-up about what they shared. "
                    f"Max 2 sentences. No lists.\n\n"
                    f"User in {region}: {message}"
                )

            elif intent == "validation":
                # User is venting — reflect back, don't advise
                prompt = (
                    f"{SENTINEL_FINE_TUNE_PROMPT}\n\n"
                    f"USER PREFERENCES: {long_term_prefs}\n"
                    f"THINKER: Sentiment={sentiment} {negation_note}. Negations={negation_count}.\n"
                    f"Recent Chat:\n{formatted_history}\n\n"
                    f"ROUTING: VALIDATION — The user needs to feel heard, not advised.\n"
                    f"RULES: Use reflective listening ONLY. Mirror their feeling back, "
                    f"then ask ONE open-ended question. Do NOT give advice. "
                    f"If negation_count is odd, interpret the FLIPPED meaning correctly. "
                    f"Max 2 sentences. No lists.\n\n"
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
                    f"FINAL INSTRUCTION: Draw from Expert Context but keep tone human and peer-to-peer. "
                    f"If negation_count is odd, address the ACTUAL meaning correctly. "
                    f"Max 3 sentences. No bullet points."
                )

            # 4. Call Gemini (with retry on rate limit)
            if ai_client:
                for i in range(3):
                    try:
                        response = await asyncio.to_thread(
                            ai_client.models.generate_content,
                            model=AI_MODEL_NAME,
                            contents=prompt
                        )
                        response_text = response.text.strip()
                        break
                    except Exception as e:
                        if "429" in str(e):
                            await asyncio.sleep((i + 1) * 3)
                        else:
                            logger.error(f"Gemini generation error: {e}")
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
            
            # Handle JSON commands (Feedback)
            if data.startswith("{"):
                try:
                    import json
                    cmd = json.loads(data)
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
