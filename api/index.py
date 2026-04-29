from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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
from profanity_check import predict

from api._database import create_user_profile, update_trust_score, get_trust_score, SessionLocal, UserProfile
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
# --- Sentinel Hidden Reasoning Layer (3-Stage Thinker Pipeline) ---
SENTINEL_THINKER_PROMPT = """
[INTERNAL MONOLOGUE - DO NOT SHOW TO USER]

STAGE 1 — CULTURAL CONTEXT CHECK (Kenyan Nuance Gate):
- Is the user invoking Kenyan-specific stressors? (e.g., Black Tax, Sapa, KCSE pressure, Hustler Fund, land disputes, chama obligations, gender-based violence, tribal/ethnic references)
- If yes: weave that cultural reality into your response — do NOT treat it as a generic Western mental-health scenario.
- Note the user's county. Is this a high-density urban area (Nairobi, Mombasa, Kisumu) or rural/peri-urban (Kiambu, Murang'a, Kwale)? Adjust tone accordingly — rural users often need more grounded, community-based framing.

STAGE 2 — REFLECTIVE LISTENING (Mirror Before Solving):
- Before ANY advice or resource, mirror the user's specific pain back to them.
- Use their exact words where possible (e.g., if they say 'I'm tired of being the strong one', reply 'It sounds like carrying everyone else's weight has left nothing for you').
- Do NOT jump to solutions. Ask one open-ended question maximum.
- Avoid: 'Have you tried...', 'You should...', 'I recommend...' — these break the peer-support contract.

STAGE 3 — RESPONSE SYNTHESIS:
- Integrate Stage 1 (cultural lens) + Stage 2 (reflective mirror) into a single, flowing human response.
- Max 3 sentences. No bullet points. No clinical language.
- End with a gentle open question OR a grounding phrase, not a directive.

[RESPONSE - SHOW TO USER]
Generate the Stage 3 synthesised response only. The user never sees the monologue above.
"""

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

def is_safe_local(text: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason). 
    Checks for PII (Personal Identifiable Info) and Malicious Intent locally.
    """
    # 1. Check for Doxing (Phone numbers)
    if re.search(r"(\+254|07|01)\d{8}", text):
        return False, "Privacy Alert: For your safety, do not share phone numbers yet."

    # 2. Check for Profanity/Aggression
    # predict() returns 1 if offensive, 0 if safe
    if predict([text])[0] == 1:
        return False, "System Alert: Let's keep our language healing and safe."

    # 3. Check for specific "Violence" keywords (Manual Override)
    danger_words = ["kill", "hurt", "attack", "stab", "blood", "die"]
    if any(word in text.lower() for word in danger_words):
        return False, "Safety Alert: It sounds like you're feeling a lot of anger right now. That's a heavy load to carry. Before you act on those feelings, would you like to speak with a professional at the Red Cross?"

    return True, ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the local AI brain once
    logger.info("Spawning Sentinel Local Brain...")
    load_expert_brain()
    # This warm-up ensures the first response is instant
    get_kenyan_fallback("Habari")
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
        Generates a compassionate AI response with:
        - Intent-Based Routing (Social / Validation / Expert)
        - 3-Stage Thinker Pipeline (Cultural → Reflective → Synthesised)
        - Human Processing Delay (1.2–2.5 s jitter) to prevent robotic cadence
        - Crisis (depth ≥ 0.7) sessions are answered immediately without delay
        """
        try:
            # 1. Retrieve Context & Memory
            db = SessionLocal()
            user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
            region = user.region if user else "Kenya"
            db.close()

            user_meta = self.user_data.get(session_id, {})
            history = user_meta.get("history", [])[-8:]
            formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])

            # --- INTENT ROUTING LOGIC ---
            user_input_clean = message.lower().strip().strip("?!.")
            greetings = ["hello", "hi", "hey", "yo", "sup", "habari", "sasa"]
            positive_words = POSITIVE_SIGNALS
            response_text = None

            # LEVEL 1: SOCIAL OVERRIDE (Greetings)
            if user_input_clean in greetings:
                response_text = f"Hey! Glad you made it to the sanctuary. I see you're connecting from {region}—how are you really doing today?"

            # LEVEL 2: POSITIVE MOOD — never route to crisis fallback
            elif any(word in user_input_clean for word in positive_words):
                response_text = f"Love that! It's always good when things are feeling {user_input_clean}. What's been making it that way?"

            elif len(user_input_clean.split()) < 3 and not is_nudge:
                vibe_responses = {
                    "bad": "I'm sorry to hear that. Sometimes the days just feel heavy. Want to tell me more?",
                    "okay": "Just 'okay' can be a lot sometimes. I'm here if you want to unpack that.",
                    "tired": "Man, I feel that. Life can really drain you. What's the biggest drain right now?",
                    "stressed": "Stress has a way of piling up. What's been sitting on your chest lately?",
                    "sad": "Sadness deserves space too. I'm right here—what's going on?",
                }
                response_text = vibe_responses.get(user_input_clean, "I'm listening. Sometimes it's hard to put words to it—take your time. What's on your mind?")

            # LEVEL 3: DEEP LAYER (Expert Brain + AI)
            if not response_text:
                # Retrieve Expert Wisdom
                expert_context = ""
                if expert_index is not None and expert_archive is not None and embedder is not None:
                    try:
                        query_vec = embedder.encode([message])
                        D, I = expert_index.search(query_vec.astype('float32'), 2)
                        for idx in I[0]:
                            if idx != -1:
                                expert_context += f"\nExpert Wisdom: {expert_archive.iloc[idx]['answerText'][:300]}..."
                    except Exception as e:
                        logger.error(f"Expert brain search error: {e}")

                # Dynamic Sentiment Tuning
                sentiment_guide = ""
                if depth < 0.3:
                    sentiment_guide = "User seems okay. Keep it high-energy, use emojis, and be a fun peer."
                else:
                    sentiment_guide = "User is struggling. Slow down, use fewer words, and validate their feelings ('I hear you, that sounds heavy') before anything else."

                # Build Thinker Prompt (merges Fine-Tune rules + Reasoning structure)
                if is_nudge:
                    prompt = f"{SENTINEL_FINE_TUNE_PROMPT}\n\nRecent Chat:\n{formatted_history}\nThe user in {region} has been silent. Ask a gentle, peer-like follow up. Max 1 sentence."
                else:
                    prompt = f"""
                    {SENTINEL_FINE_TUNE_PROMPT}
                    {SENTINEL_THINKER_PROMPT}
                    
                    SENTIMENT GUIDE: {sentiment_guide}
                    EXPERT CONTEXT: {expert_context}
                    
                    Recent Chat Memory:
                    {formatted_history}
                    
                    User in {region}: {message}
                    """
                
                # Call AI with Fallback
                if ai_client:
                    for i in range(3):
                        try:
                            response = await asyncio.to_thread(
                                ai_client.models.generate_content,
                                model=AI_MODEL_NAME,
                                contents=prompt
                            )
                            raw = response.text
                            # Strip hidden [INTERNAL MONOLOGUE] — user only sees [RESPONSE]
                            if "[RESPONSE - SHOW TO USER]" in raw:
                                response_text = raw.split("[RESPONSE - SHOW TO USER]")[1].strip()
                            elif "RESPONSE:" in raw:
                                response_text = raw.split("RESPONSE:")[1].strip()
                            else:
                                response_text = raw
                            break
                        except Exception as e:
                            if "429" in str(e):
                                await asyncio.sleep((i + 1) * 3)
                            else:
                                break

            # FINAL FALLBACK (If level 3 fails or wasn't triggered)
            if not response_text:
                response_text = get_kenyan_fallback(message)

            # ── PULSE LOGIC METER: Human Processing Delay ─────────────────────
            # Crisis inputs (depth ≥ 0.7) are answered immediately.
            # Everything else gets a 1.2–2.5 s jitter to simulate genuine
            # human thinking cadence and prevent a robotic "instant" reply.
            if depth < 0.7 and not is_nudge:
                await asyncio.sleep(random.uniform(1.2, 2.5))

            # 6. Update Memory & Send
            if session_id in self.user_data:
                self.user_data[session_id]["history"].append({"role": "User", "content": message})
                self.user_data[session_id]["history"].append({"role": "Sentinel", "content": response_text})

            if session_id in self.active_connections:
                await self.active_connections[session_id].send_json({
                    "type": "peer",
                    "content": f"[Sentinel]: {response_text}",
                    "timestamp": str(datetime.datetime.now())
                })
        except Exception as e:
            logger.error(f"AI Error: {e}")

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
