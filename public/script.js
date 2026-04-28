/* ═══════════════════════════════════════════════════════
   GATEWAY — Curtain transition logic
   ═══════════════════════════════════════════════════════ */

let socketInitialized = false;
let gatewaySocket     = null;   // WS reference held here until sanctuary init

document.getElementById('welcome-btn').addEventListener('click', () => {
    const btn = document.getElementById('welcome-btn');

    // Prevent double-clicks while connecting
    if (btn.disabled) return;
    btn.disabled  = true;
    btn.textContent = 'Connecting...';
    btn.style.opacity = '0.7';

    // Open the WebSocket NOW — curtain waits for onopen
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    gatewaySocket   = new WebSocket(`${protocol}//${window.location.host}/ws`);

    // SUCCESS — handshake confirmed → open the curtain
    gatewaySocket.onopen = () => {
        const showcase = document.getElementById('showcase');
        showcase.classList.add('open');

        // Reveal the chat sanctuary once curtains finish sliding (1.5 s)
        setTimeout(() => {
            const sanctuary = document.getElementById('sanctuary-root');
            sanctuary.classList.add('visible');
            sanctuary.removeAttribute('aria-hidden');

            if (!socketInitialized) {
                socketInitialized = true;
                initSanctuary(gatewaySocket);   // hand the live socket over
            }
        }, 1500);
    };

    // FAILURE — let the user try again
    gatewaySocket.onerror = () => {
        btn.disabled    = false;
        btn.textContent = 'Welcome to e-motions';
        btn.style.opacity = '1';
        // Brief visual feedback on the button
        btn.style.borderColor = '#ff4d4d';
        btn.style.color       = '#ff4d4d';
        setTimeout(() => {
            btn.style.borderColor = '';
            btn.style.color       = '';
        }, 2000);
    };
});


/* ═══════════════════════════════════════════════════════
   SANCTUARY — Chat & WebSocket logic
   ═══════════════════════════════════════════════════════ */

function initSanctuary(existingSocket) {
    // Reuse the socket that was opened during the handshake check
    const socket = existingSocket;

    let safeExitContact = "1199"; // Default Red Cross
    let currentMode = "waiting";  // 'waiting' (AI) or 'paired' (Human)
    let inactivityTimer;
    let thinkingBubble = null;

    const diaryFeed = document.getElementById('diary-feed');
    const input = document.getElementById('entry-input');
    const statusText = document.getElementById('status-text');
    const statusIndicator = document.querySelector('.status-indicator');
    const logicMeter = document.getElementById('logic-meter');
    const sosBtn = document.getElementById('sos-btn');
    const sendBtn = document.getElementById('send-btn');

    function showThinking() {
        logicMeter.classList.add('active');
        if (!thinkingBubble) {
            thinkingBubble = document.createElement('div');
            thinkingBubble.className = 'thinking-bubble';
            thinkingBubble.innerHTML = '<span>●</span><span>●</span><span>●</span>';
            diaryFeed.appendChild(thinkingBubble);
            diaryFeed.scrollTop = diaryFeed.scrollHeight;
        }
    }

    function hideThinking() {
        logicMeter.classList.remove('active');
        if (thinkingBubble) {
            thinkingBubble.remove();
            thinkingBubble = null;
        }
    }

    function log(message, type = 'system') {
        const entry = document.createElement('div');
        entry.className = `log-entry ${type} system-note`;
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        entry.textContent = `[${timestamp}] ${message}`;
        diaryFeed.appendChild(entry);
        diaryFeed.scrollTop = diaryFeed.scrollHeight;
    }

    function startInactivityTimer() {
        clearTimeout(inactivityTimer);
        inactivityTimer = setTimeout(() => {
            if (currentMode === "waiting" && socket && socket.readyState === WebSocket.OPEN) {
                log("Still here. Just listening...", "system");
                socket.send("__TRIGGER_AI_NUDGE__");
            }
        }, 30000);
    }

    function renderMessage(content, type) {
        const div = document.createElement('div');
        div.className = `msg ${type}-msg`;
        div.textContent = content;
        diaryFeed.appendChild(div);
        diaryFeed.scrollTop = diaryFeed.scrollHeight;
    }

    function handleSend() {
        const text = input.value.trim();
        if (text) {
            socket.send(text);
            renderMessage(text, 'my');
            input.value = '';
            input.style.height = 'auto';
            startInactivityTimer();
            if (currentMode === 'waiting') showThinking();
        }
    }

    function triggerSOS() {
        if (confirm(`Connecting you to Regional Support (${safeExitContact}). Proceed?`)) {
            const phoneMatch = safeExitContact.match(/\d+/);
            const phone = phoneMatch ? phoneMatch[0] : "1199";
            window.location.href = `tel:${phone}`;
        }
    }

    // ── Socket Events ──────────────────────────────────────
    // The socket is already OPEN (handshake completed before curtain opened),
    // so we update the UI state immediately rather than waiting for onopen.
    statusText.textContent = "Connected to Safe Zone";
    statusIndicator.classList.add('connected');
    startInactivityTimer();

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "system") {
            if (data.content.includes("Connected to a peer") || data.content.includes("A fellow traveler has joined")) {
                currentMode = "paired";
                statusIndicator.style.background = "rgba(76, 175, 80, 0.2)";
                document.getElementById('status-dot').style.background = "#4caf50";
            } else if (data.content.includes("Sentinel AI")) {
                currentMode = "waiting";
                statusIndicator.style.background = "rgba(168, 85, 247, 0.2)";
                document.getElementById('status-dot').style.background = "#a855f7";
            }

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
            }
        } else {
            hideThinking();
            renderMessage(data.content, data.type);
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

    // ── UI Listeners ───────────────────────────────────────
    if (sendBtn) sendBtn.addEventListener('click', handleSend);
    if (sosBtn) sosBtn.addEventListener('click', triggerSOS);

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
}
