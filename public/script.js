const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

let safeExitContact = "1199"; // Default Red Cross

const diaryFeed = document.getElementById('diary-feed');
const input = document.getElementById('entry-input');
const statusText = document.getElementById('status-text');
const statusIndicator = document.querySelector('.status-indicator');

function log(message, type = 'system') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type} system-note`;
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    entry.textContent = `[${timestamp}] ${message}`;
    diaryFeed.appendChild(entry);
    diaryFeed.scrollTop = diaryFeed.scrollHeight;
}

let currentMode = "waiting"; // 'waiting' (AI) or 'paired' (Human)

let inactivityTimer;

function startInactivityTimer() {
    clearTimeout(inactivityTimer);
    // If user is silent for 30 seconds and alone
    inactivityTimer = setTimeout(() => {
        if (currentMode === "waiting" && socket && socket.readyState === WebSocket.OPEN) {
            log("Still here. Just listening...", "system");
            socket.send("__TRIGGER_AI_NUDGE__");
        }
    }, 30000);
}

socket.onopen = () => {
    statusText.textContent = "Connecting to Safe Zone...";
    statusIndicator.classList.add('connected');
    startInactivityTimer();
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // Detect Handover or Peer Connection
    if (data.type === "system") {
        if (data.content.includes("Connected to a peer") || data.content.includes("A fellow traveler has joined")) {
            currentMode = "paired";
            statusIndicator.style.background = "rgba(76, 175, 80, 0.2)"; // Green glass for Human
            document.getElementById('status-dot').style.background = "#4caf50";
        } else if (data.content.includes("Sentinel AI")) {
            currentMode = "waiting";
            statusIndicator.style.background = "rgba(168, 85, 247, 0.2)"; // Purple glass for AI
            document.getElementById('status-dot').style.background = "#a855f7";
        }
        
        // Update Safe Zone text if location is mentioned
        if (data.content.includes("recognized you're in")) {
            const parts = data.content.split("recognized you're in ");
            if (parts.length > 1) {
                statusText.textContent = "Safe Zone: " + parts[1].split(".")[0];
            }
        }
        
        log(data.content, "system");
    } else if (data.type === "metadata") {
        if (data.key === "safe_exit_contact") {
            safeExitContact = data.value;
            console.log("Updated Safe Exit Contact:", safeExitContact);
        }
    } else {
        renderMessage(data.content, data.type);
        // Ensure Sentinel messages also trigger purple if not already set
        if (data.content.startsWith("[Sentinel]")) {
            currentMode = "waiting";
            statusIndicator.style.background = "rgba(168, 85, 247, 0.2)";
            document.getElementById('status-dot').style.background = "#a855f7";
        }
    }
};

socket.onclose = () => {
    statusText.textContent = "Disconnected";
    statusIndicator.classList.remove('connected');
    clearTimeout(inactivityTimer);
};

function handleSend() {
    const text = input.value.trim();
    if (text) {
        socket.send(text);
        renderMessage(text, 'my');
        input.value = '';
        input.style.height = 'auto';
        startInactivityTimer(); // Reset timer on every message
    }
}


// Auto-expand textarea
input.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

function renderMessage(content, type) {
    const div = document.createElement('div');
    div.className = `msg ${type}-msg`;
    div.textContent = content;
    diaryFeed.appendChild(div);
    diaryFeed.scrollTop = diaryFeed.scrollHeight;
}

function triggerSOS() {
    const displayContact = safeExitContact.includes(":") ? safeExitContact.split(":")[1].trim() : safeExitContact;
    if (confirm(`Connecting you to Regional Support (${safeExitContact}). Proceed?`)) {
        // Extract phone number if it's in format "Name: Number"
        const phone = safeExitContact.match(/\d+/) ? safeExitContact.match(/\d+/)[0] : "1199";
        window.location.href = `tel:${phone}`;
    }
}
