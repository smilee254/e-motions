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

from api.database import create_user_profile, update_trust_score, get_trust_score, SessionLocal, UserProfile
from api.fallback import (
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
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    # Modern 2026 SDK Initialization
    ai_client = genai.Client(api_key=GEMINI_KEY)
    # Target Model: Verified stable 2.5-flash
    AI_MODEL_NAME = "gemini-2.5-flash"
else:
    ai_client = None

# --- Local GeoIP Configuration ---
GEOIP_DB_PATH = os.path.join("api", "dbip-city-lite.mmdb")
try:
    geoip_reader = geoip2.database.Reader(GEOIP_DB_PATH)
except Exception as e:
    logger.error(f"Failed to load GeoIP database: {e}")
    geoip_reader = None

# --- Sentinel Expert Brain (CounselChat) ---
EXPERT_INDEX_PATH = os.path.join("api", "sentinel_brain.index")
EXPERT_PKL_PATH = os.path.join("api", "expert_archive.pkl")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        # lobby: {region: [session_id1, session_id2]}
        self.lobby: Dict[str, List[str]] = {}
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

        # Start the matching process in the background
        # Also enable AI immediately so they aren't left in silence while waiting
        self.ai_sessions.add(session_id)
        asyncio.create_task(self.match_user(session_id, sub_county, county))
        return session_id

    @staticmethod
    def _room_key(location: str) -> str:
        """Normalises any location string to a consistent lobby key."""
        return re.sub(r'[^a-z0-9]+', '_', location.lower().strip()).strip('_')

    async def match_user(self, session_id: str, sub_county: str, county: str):
        """Tiered matching algorithm: Sub-County -> County -> National -> Sentinel AI"""
        
        # ── Normalise room keys for consistent lobby lookups ──────────────
        key_sub   = self._room_key(sub_county)   # e.g. "room_kiambu_ruiru"
        key_county = self._room_key(county)       # e.g. "room_kiambu"
        key_national = "room_kenya_national"

        # Tier 1: Sub-County room (30 s)
        self.lobby.setdefault(key_sub, [])
        if self.lobby[key_sub]:
            peer_id = self.lobby[key_sub].pop(0)
            await self.pair_users(session_id, peer_id, f"local peer from {sub_county}")
            return

        self.lobby[key_sub].append(session_id)
        logger.info(f"[LOBBY] {session_id} waiting in room_{key_sub} (Tier 1)")
        await asyncio.sleep(30)

        if session_id in self.lobby.get(key_sub, []):
            self.lobby[key_sub].remove(session_id)

            # Tier 2: County room (30 s)
            await self.send_system_msg(
                session_id,
                f"Still looking for a local peer in {sub_county}... expanding to {county} County."
            )
            self.lobby.setdefault(key_county, [])
            if self.lobby[key_county]:
                peer_id = self.lobby[key_county].pop(0)
                await self.pair_users(session_id, peer_id, f"county neighbor from {county}")
                return

            self.lobby[key_county].append(session_id)
            logger.info(f"[LOBBY] {session_id} waiting in room_{key_county} (Tier 2)")
            await asyncio.sleep(30)

            if session_id in self.lobby.get(key_county, []):
                self.lobby[key_county].remove(session_id)

                # Tier 3: National room (30 s)
                await self.send_system_msg(
                    session_id,
                    "Still quiet in your county... opening the sanctuary to all of Kenya."
                )
                self.lobby.setdefault(key_national, [])
                if self.lobby[key_national]:
                    peer_id = self.lobby[key_national].pop(0)
                    await self.pair_users(session_id, peer_id, "national peer")
                    return

                self.lobby[key_national].append(session_id)
                logger.info(f"[LOBBY] {session_id} waiting in room_{key_national} (Tier 3)")
                await asyncio.sleep(30)

            # Tier 4: Sentinel AI — clean up all lobbies first
            for key in [key_sub, key_county, key_national]:
                if key in self.lobby and session_id in self.lobby[key]:
                    self.lobby[key].remove(session_id)

            if session_id not in self.matches:
                self.ai_sessions.add(session_id)
                await self.send_system_msg(
                    session_id,
                    "The National Sanctuary is holding space for you. Sentinel AI is here — take your time."
                )
                grounding = get_regional_grounding(county)
                await self.send_system_msg(session_id, grounding)
                logger.info(f"[LOBBY] {session_id} routed to Sentinel AI (Tier 4 — National Sanctuary)")

    async def pair_users(self, id1: str, id2: str, level: str):
        if id1 in self.ai_sessions:
            self.ai_sessions.remove(id1)
        if id2 in self.ai_sessions:
            self.ai_sessions.remove(id2)

        self.matches[id1] = id2
        self.matches[id2] = id1
        
        await self.send_system_msg(id1, f"Connected to a {level}. Your diary is listening.")
        await self.send_system_msg(id2, f"Connected to a {level}. Your diary is listening.")

    def disconnect(self, session_id: str):
        peer_id = self.matches.get(session_id)
        
        # Cleanup matches
        self.matches.pop(session_id, None)
        if peer_id:
            self.matches.pop(peer_id, None)
            
        # Cleanup AI sessions
        self.ai_sessions.discard(session_id)
            
        # Cleanup lobby
        for loc in self.lobby:
            if session_id in self.lobby[loc]:
                self.lobby[loc] = [s for s in self.lobby[loc] if s != session_id]
        
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

            # LEVEL 2b: ASSESSMENT LAYER (Short Vague Inputs)
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
            # Check if there's a high-depth match available in the lobby now
            if depth > 0.5:
                # Try to find a human match who is also high depth or just available
                await self.try_priority_match(sender_id)
            
            if sender_id in self.ai_sessions: # Might have been matched above
                await self.handle_ai_chat(sender_id, message, depth=depth)

    async def try_priority_match(self, session_id: str):
        """
        Crisis preemption: when a user's depth ≥ 0.5, promote them to the
        front of any available regional lobby before falling through to AI.
        Checks normalised room keys in order: sub-county → county → national.
        """
        data = self.user_data.get(session_id)
        if not data:
            return

        county    = data["county"]
        sub_county = data["sub_county"]

        for loc, label in [
            (self._room_key(sub_county), "priority local peer"),
            (self._room_key(county),     "priority regional peer"),
            ("room_kenya_national",       "priority national peer"),
        ]:
            if self.lobby.get(loc):
                peer_id = self.lobby[loc].pop(0)
                await self.pair_users(session_id, peer_id, label)
                logger.info(f"[CRISIS] {session_id} preemptively matched to {peer_id} via {loc}")
                return

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

# Mount static files at root AFTER routes are defined
app.mount("/", StaticFiles(directory="public", html=True), name="public")
