// ---------------------------------------------------------------------------
// @vapi-ai/web has no plain <script src> CDN build -- it's published for
// bundlers, not global-script use. esm.sh serves any npm package as a
// browser-native ES module on demand, which is why this file is loaded as
// type="module" in index.html and imports directly like this.
// ---------------------------------------------------------------------------
import Vapi from "https://esm.sh/@vapi-ai/web@2.6.1";

// ---------------------------------------------------------------------------
// Configuration — fill these in before deploying.
// ---------------------------------------------------------------------------
const CONFIG = {
  // Your deployed FastAPI backend's base URL (the Vercel URL from /api).
  // Used here only for the text-input fallback -- Vapi calls this same
  // endpoint directly during real voice calls, this app doesn't proxy that.
  API_BASE_URL: "https://your-backend.vercel.app",

  // From your Vapi dashboard: Settings -> API Keys -> Public Key.
  VAPI_PUBLIC_KEY: "YOUR_VAPI_PUBLIC_KEY",

  // The assistant ID you create in the Vapi dashboard, configured with
  // Custom LLM pointing at this same backend's /chat/completions endpoint.
  VAPI_ASSISTANT_ID: "YOUR_VAPI_ASSISTANT_ID",
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentMode = "hr_policy"; // "hr_policy" | "interview_prep"
let vapiClient = null;
let isCallActive = false;

const modeLabels = {
  hr_policy: "HR Policy",
  interview_prep: "Interview Prep",
};

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const chatArea = document.getElementById("chat-area");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const micLabel = document.getElementById("mic-label");
const currentModeValue = document.getElementById("current-mode-value");
const kbActiveLabel = document.getElementById("kb-active-label");
const modeButtons = document.querySelectorAll(".mode-btn");
const domainCards = document.querySelectorAll(".domain-card");

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------
function setMode(mode) {
  currentMode = mode;
  modeButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  currentModeValue.textContent = modeLabels[mode];
  kbActiveLabel.textContent = modeLabels[mode];
}

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

domainCards.forEach((card) => {
  card.addEventListener("click", () => {
    setMode("interview_prep");
    const domainName = card.querySelector(".domain-title").textContent;
    textInput.value = `What should I expect for a ${domainName} interview?`;
    textInput.focus();
    sendBtn.disabled = false;
  });
});

// ---------------------------------------------------------------------------
// Chat rendering
// ---------------------------------------------------------------------------
function appendMessage(role, text) {
  const row = document.createElement("div");
  row.className = role === "user" ? "chat-row user" : "chat-row";

  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.innerHTML = role === "user"
    ? '<i class="ti ti-user"></i>'
    : '<i class="ti ti-microphone"></i>';

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = text;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatArea.appendChild(row);
  chatArea.scrollTop = chatArea.scrollHeight;

  return bubble;
}

// ---------------------------------------------------------------------------
// Text-input fallback -- calls the backend directly for a quick text test
// without needing a live Vapi call.
// ---------------------------------------------------------------------------
async function sendTextMessage() {
  const query = textInput.value.trim();
  if (!query) return;

  appendMessage("user", query);
  textInput.value = "";
  sendBtn.disabled = true;

  const loadingBubble = appendMessage("assistant", "Thinking...");

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: query }],
      }),
    });
    const data = await response.json();
    const answer = data?.choices?.[0]?.message?.content || "Sorry, I couldn't get a response.";
    loadingBubble.textContent = answer;
  } catch (err) {
    loadingBubble.textContent = "Couldn't reach the backend. Check that API_BASE_URL in app.js points to your deployed server.";
  }
}

sendBtn.addEventListener("click", sendTextMessage);
textInput.addEventListener("input", () => {
  sendBtn.disabled = textInput.value.trim().length === 0;
});
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !sendBtn.disabled) sendTextMessage();
});

// ---------------------------------------------------------------------------
// Voice (Vapi Web SDK) -- hold-to-speak
// ---------------------------------------------------------------------------
function initVapi() {
  vapiClient = new Vapi(CONFIG.VAPI_PUBLIC_KEY);

  vapiClient.on("call-start", () => {
    isCallActive = true;
    micBtn.classList.add("recording");
    micLabel.textContent = "Listening...";
  });

  vapiClient.on("call-end", () => {
    isCallActive = false;
    micBtn.classList.remove("recording");
    micLabel.textContent = "Hold to speak";
  });

  vapiClient.on("message", (message) => {
    if (message.type === "transcript" && message.transcriptType === "final") {
      appendMessage(message.role === "user" ? "user" : "assistant", message.transcript);
    }
  });
}

function startCall() {
  if (!vapiClient) return;
  vapiClient.start(CONFIG.VAPI_ASSISTANT_ID);
}

function stopCall() {
  if (!vapiClient || !isCallActive) return;
  vapiClient.stop();
}

// Hold-to-speak: press and hold the mic button to talk, release to stop.
micBtn.addEventListener("mousedown", startCall);
micBtn.addEventListener("mouseup", stopCall);
micBtn.addEventListener("mouseleave", () => {
  if (isCallActive) stopCall();
});
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startCall(); });
micBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopCall(); });

initVapi();