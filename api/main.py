from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
import uuid
import datetime
import re
import httpx
import os
from api.database import create_user_profile, update_trust_score, get_trust_score
from profanity_check import predict

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

app = FastAPI(title="e-motions API")

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
            peer_id = self.lobby[region].pop(0)
            self.matches[session_id] = peer_id
            self.matches[peer_id] = session_id
            
            await self.send_system_msg(session_id, "Match found. Your diary is listening.")
            await self.send_system_msg(peer_id, "Match found. Your diary is listening.")
        else:
            self.lobby[region].append(session_id)
            await self.send_system_msg(session_id, f"Waiting for an audience in {region}...")
        
        return session_id

    def disconnect(self, session_id: str):
        peer_id = self.matches.get(session_id)
        
        # Cleanup matches
        self.matches.pop(session_id, None)
        if peer_id:
            self.matches.pop(peer_id, None)
            
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

    async def relay_message(self, sender_id: str, message: str):
        peer_id = self.matches.get(sender_id)
        if peer_id and peer_id in self.active_connections:
            await self.active_connections[peer_id].send_json({
                "type": "peer",
                "content": message,
                "timestamp": str(datetime.datetime.now())
            })

manager = ConnectionManager()

@app.websocket("/ws/{region}")
async def websocket_endpoint(websocket: WebSocket, region: str):
    session_id = await manager.connect(websocket, region)
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
                    break # Optional: Terminate connection
                continue
            
            if session_id in manager.matches:
                await manager.relay_message(session_id, data)
            else:
                await manager.send_system_msg(session_id, "Writing into the void... Still looking for a listener.")
                
    except WebSocketDisconnect:
        peer_id = manager.disconnect(session_id)
        if peer_id:
            await manager.send_system_msg(peer_id, "Your peer has disconnected. Finding a new audience...")

# Mount static files at root AFTER routes are defined
app.mount("/", StaticFiles(directory="public", html=True), name="public")
