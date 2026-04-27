from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Set, cast, Any
import uuid
import datetime
import re
import os
from google import genai
import logging
import asyncio
from dotenv import load_dotenv
from api.database import create_user_profile, update_trust_score, get_trust_score, SessionLocal, UserProfile
from profanity_check import predict
from api.fallback import get_kenyan_fallback, get_regional_grounding, detect_depth, REGIONAL_CONTACTS
import geoip2.database
from fastapi import Request


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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the local AI brain once
    logger.info("Spawning Sentinel Local Brain...")
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
            "last_msg": ""
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

    async def match_user(self, session_id: str, sub_county: str, county: str):
        """Tiered matching algorithm: Sub-County -> County -> Sentinel AI"""
        
        # Tier 1: Sub-County (30 seconds)
        if sub_county not in self.lobby:
            self.lobby[sub_county] = []
        
        # Check if anyone is waiting in the sub-county lobby
        if self.lobby[sub_county]:
            peer_id = self.lobby[sub_county].pop(0)
            await self.pair_users(session_id, peer_id, "local peer")
            return

        # No one found immediately, add to sub-county lobby
        self.lobby[sub_county].append(session_id)
        
        # Wait 30 seconds for local peer
        await asyncio.sleep(30)
        
        # Check if still in sub-county lobby (i.e., not matched yet)
        if session_id in self.lobby.get(sub_county, []):
            self.lobby[sub_county].remove(session_id)
            
            # Tier 2: County (Next 30 seconds)
            await self.send_system_msg(session_id, f"Still looking for a local peer in {sub_county}... expanding to {county} County.")
            
            if county not in self.lobby:
                self.lobby[county] = []
            
            if self.lobby[county]:
                peer_id = self.lobby[county].pop(0)
                await self.pair_users(session_id, peer_id, "county neighbor")
                return
            
            self.lobby[county].append(session_id)
            
            # Wait another 30 seconds
            await asyncio.sleep(30)
            
            # Tier 3: National (Next 30 seconds)
            if session_id in self.lobby.get(county, []):
                self.lobby[county].remove(session_id)
                
                await self.send_system_msg(session_id, "Still quiet in your county... opening the sanctuary to all of Kenya.")
                
                if "Kenya" not in self.lobby:
                    self.lobby["Kenya"] = []
                
                if self.lobby["Kenya"]:
                    peer_id = self.lobby["Kenya"].pop(0)
                    await self.pair_users(session_id, peer_id, "national peer")
                    return
                
                self.lobby["Kenya"].append(session_id)
                
                # Wait final 30 seconds
                await asyncio.sleep(30)

            # Tier 4: Sentinel AI Fallback
            # Check all possible lobbies to be sure
            for loc in [sub_county, county, "Kenya"]:
                if loc in self.lobby and session_id in self.lobby[loc]:
                    self.lobby[loc].remove(session_id)
            
            # If still unmatched (not in self.matches), trigger AI
            if session_id not in self.matches:
                self.ai_sessions.add(session_id)
                await self.send_system_msg(session_id, "The sanctuary is quiet across the country. Sentinel AI is here to listen to you.")
                # Sentinel grounding
                grounding = get_regional_grounding(county)
                await self.send_system_msg(session_id, grounding)

    async def pair_users(self, id1: str, id2: str, level: str):
        if id1 in self.ai_sessions: self.ai_sessions.remove(id1)
        if id2 in self.ai_sessions: self.ai_sessions.remove(id2)

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
        """Generates a compassionate AI response with Exponential Backoff and Local Fallback."""
        try:
            # Get user's region for grounding
            db = SessionLocal()
            user = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
            region = user.region if user else "Kenya"
            db.close()

            if is_nudge:
                prompt = f"The user in {region} has been silent. Ask a gentle, open-ended question. Max 2 sentences."
            else:
                prompt = f"""
                You are 'The Sanctuary', a compassionate listener.
                User is in: {region}
                Intensity/Depth: {depth}
                Tone: Calm, minimal, non-judgmental. No medical advice.
                Constraint: Max 2 sentences. Include local grounding if depth > 0.5.
                User: {message}
                """
            
            # Implementation of the Exponential Backoff with Fallback (Modern SDK)
            response_text = None
            if ai_client:
                for i in range(3): # Initial + 2 retries
                    try:
                        # Using asyncio.to_thread for the modern sync client
                        response = await asyncio.to_thread(
                            ai_client.models.generate_content,
                            model=AI_MODEL_NAME,
                            contents=prompt
                        )
                        response_text = response.text
                        break
                    except Exception as e:
                        if "429" in str(e):
                            wait = (i + 1) * 3
                            logger.warning(f"Quota hit. Retrying in {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            logger.warning(f"AI API Error: {e}. Switching to local fallback.")
                            break
            else:
                logger.info("Sentinel AI client missing. Using local fallback.")

            # If Gemini failed, use the Kenyan Local Fallback
            if not response_text:
                logger.info("Triggering Local Kenyan Fallback.")
                if is_nudge:
                    response_text = "Still here. Just listening..."
                else:
                    response_text = get_kenyan_fallback(message)

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
        """Attempts to find a human peer for a high-depth user."""
        data = self.user_data.get(session_id)
        if not data: return
        
        county = data["county"]
        # Look in county lobby for anyone
        if self.lobby.get(county):
            peer_id = self.lobby[county].pop(0)
            await self.pair_users(session_id, peer_id, "priority regional peer")
            return
        
        # Also check sub-county (though unlikely if not in county)
        sub_county = data["sub_county"]
        if self.lobby.get(sub_county):
            peer_id = self.lobby[sub_county].pop(0)
            await self.pair_users(session_id, peer_id, "priority local peer")
            return

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Detect IP
    client_ip = websocket.client.host
    # If behind a proxy (like Vercel/Cloudflare), we might need X-Forwarded-For
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0]
        
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

@app.get("/api/location")
async def get_location(request: Request):
    # Middleware already populated request.state.geo
    return getattr(request.state, "geo", {"city": "Unknown", "county": "Unknown", "country": "Kenya"})

# Mount static files at root AFTER routes are defined
app.mount("/", StaticFiles(directory="public", html=True), name="public")
