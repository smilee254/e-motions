const region = "Nairobi"; // This could be fetched from a dropdown or IP logic later
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const socket = new WebSocket(`${protocol}//${window.location.host}/ws/${region}`);

const diaryFeed = document.getElementById('diary-feed');
const input = document.getElementById('entry-input');
const statusText = document.getElementById('status-text');
const statusIndicator = document.querySelector('.status-indicator');

socket.onopen = () => {
    statusText.textContent = "Safe Zone: " + region;
    statusIndicator.classList.add('connected');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderMessage(data.content, data.type);
};

socket.onclose = () => {
    statusText.textContent = "Disconnected";
    statusIndicator.classList.remove('connected');
};

function handleSend() {
    const text = input.value.trim();
    if (text) {
        socket.send(text);
        renderMessage(text, 'my');
        input.value = '';
        input.style.height = 'auto';
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
    if (confirm("Connecting you to Kenya Red Cross (1199). Proceed?")) {
        window.location.href = "tel:1199";
    }
}
