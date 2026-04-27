from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Set, cast, Any, Coroutine
import uuid
import datetime
import re
import httpx
import os
from google import genai
import logging
import asyncio
from dotenv import load_dotenv
from api.database import create_user_profile, update_trust_score, get_trust_score
from profanity_check import predict
from api.fallback import get_kenyan_fallback


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

app = FastAPI(title="e-motions API", lifespan=lifespan)

# Allow Vercel/Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        # active_connections: {session_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # lobby: {region: [session_id1, session_id2]}
        self.lobby: Dict[str, List[str]] = {}
        # matches: {session_id: peer_session_id}
        self.matches: Dict[str, str] = {}
        # Track who is currently talking to AI
        self.ai_sessions: Set[str] = set()

    async def connect(self, websocket: WebSocket, region: str):
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        
        # Initialize Trust Profile
        create_user_profile(session_id, region)
        
        # Regional Matching
        if region not in self.lobby:
            self.lobby[region] = []

        if self.lobby[region]:
            # HUMAN FOUND: Perform Handover
            peer_id = self.lobby[region].pop(0)
            
            # If the peer was talking to AI, stop that session
            if peer_id in self.ai_sessions:
                self.ai_sessions.remove(peer_id)
                await self.send_system_msg(peer_id, "A fellow traveler has joined. I will step back and let you two talk.")

            self.matches[session_id] = peer_id
            self.matches[peer_id] = session_id
            
            await self.send_system_msg(session_id, "Connected to a peer in your region. Your diary is listening.")
            await self.send_system_msg(peer_id, "A new soul has entered the sanctuary.")
        else:
            # NO HUMAN: Enable AI Mode immediately
            self.lobby[region].append(session_id)
            self.ai_sessions.add(session_id)
            await self.send_system_msg(session_id, "The sanctuary is quiet right now. I am the Sentinel AI; I'll listen until someone joins.")
        
        return session_id

    def disconnect(self, session_id: str):
        peer_id = self.matches.get(session_id)
        
        # Cleanup matches
        self.matches.pop(session_id, None)
        if peer_id:
            self.matches.pop(peer_id, None)
            
        # Cleanup AI sessions
        self.ai_sessions.discard(session_id)
            
        # Cleanup lobby
        for region in self.lobby:
            if session_id in self.lobby[region]:
                self.lobby[region] = [s for s in self.lobby[region] if s != session_id]
        
        self.active_connections.pop(session_id, None)
            
        return peer_id

    async def send_system_msg(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "system",
                "content": message,
                "timestamp": str(datetime.datetime.now())
            })

    async def handle_ai_chat(self, session_id: str, message: str, is_nudge: bool = False):
        """Generates a compassionate AI response with Exponential Backoff and Local Fallback."""
        if not ai_client:
            return
            
        try:
            if is_nudge:
                prompt = "The user has been silent in the sanctuary. Ask a gentle, open-ended question to help them start sharing. Max 2 sentences."
            else:
                prompt = f"""
                You are 'The Sanctuary', a compassionate listener.
                Tone: Calm, minimal, non-judgmental. No medical advice.
                Constraint: Max 2 sentences.
                User: {message}
                """
            
            # Implementation of the Exponential Backoff with Fallback (Modern SDK)
            response_text = None
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
        # Case 1: Peer Match exists
        if sender_id in self.matches:
            peer_id = self.matches[sender_id]
            if peer_id in self.active_connections:
                await self.active_connections[peer_id].send_json({
                    "type": "peer",
                    "content": message,
                    "timestamp": str(datetime.datetime.now())
                })
        
        # Case 2: User is talking to AI
        elif sender_id in self.ai_sessions:
            await self.handle_ai_chat(sender_id, message)

manager = ConnectionManager()

@app.websocket("/ws/{region}")
async def websocket_endpoint(websocket: WebSocket, region: str):
    session_id: str = cast(str, await manager.connect(websocket, region))
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
