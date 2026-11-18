"use strict";

const POLL_INTERVAL_MS = 300;
const SPEAKER_COLORS = { 1: "#2878f0", 2: "#20b45a", 3: "#ff9818" };
const WAVE_HEIGHTS = [7, 12, 9, 18, 13, 22, 16, 10, 19, 14, 8, 12, 7, 10, 16, 12, 8, 11, 7, 9];

const elements = {
  screen: document.getElementById("device-screen"),
  modelButton: document.getElementById("model-button"),
  modelToggle: document.getElementById("model-toggle"),
  sttButton: document.getElementById("stt-button"),
  sttLabel: document.getElementById("stt-label"),
  resetButton: document.getElementById("reset-button"),
  settingsButton: document.getElementById("settings-button"),
  settingsClose: document.getElementById("settings-close"),
  settingsPopover: document.getElementById("settings-popover"),
  largeTextInput: document.getElementById("large-text-input"),
  cameraStream: document.getElementById("camera-stream"),
  cameraPlaceholder: document.getElementById("camera-placeholder"),
  cameraMessage: document.getElementById("camera-message"),
  cameraLight: document.getElementById("camera-light"),
  faceLayer: document.getElementById("face-layer"),
  captionList: document.getElementById("caption-list"),
  recognitionIndicator: document.getElementById("recognition-indicator"),
  recognitionLabel: document.getElementById("recognition-label"),
  microphoneDot: document.getElementById("microphone-dot"),
  microphoneStatus: document.getElementById("microphone-status"),
  cameraDot: document.getElementById("camera-dot"),
  cameraStatus: document.getElementById("camera-status"),
  runtimeDot: document.getElementById("runtime-dot"),
  runtimeStatus: document.getElementById("runtime-status"),
  errorToast: document.getElementById("error-toast"),
};

let cameraStreamStarted = false;
let pollTimer = null;

function fitScreen() {
  const scale = Math.min(window.innerWidth / 1024, window.innerHeight / 600);
  document.documentElement.style.setProperty("--screen-scale", String(scale));
}

function initializeWaveforms() {
  document.querySelectorAll(".waveform").forEach((waveform) => {
    WAVE_HEIGHTS.forEach((height, index) => {
      const bar = document.createElement("i");
      bar.className = "wave-bar";
      bar.style.setProperty("--bar-height", `${height}px`);
      bar.style.setProperty("--bar-delay", `${index * 55}ms`);
      waveform.appendChild(bar);
    });
  });
}

function speakerName(speakers, speakerId) {
  return speakers.find((speaker) => Number(speaker.id) === Number(speakerId))?.name || `Speaker ${speakerId}`;
}

function createSpeakerSlot(speakers, speakerId) {
  if (Number(speakerId)) return createSpeakerTag(speakers, Number(speakerId));
  const blank = document.createElement("span");
  blank.className = "speaker-tag blank";
  blank.setAttribute("aria-hidden", "true");
  return blank;
}

function createSpeakerTag(speakers, speakerId) {
  const tag = document.createElement("span");
  tag.className = "speaker-tag";
  tag.textContent = speakerName(speakers, speakerId);
  tag.style.backgroundColor = SPEAKER_COLORS[speakerId] || "#8b93a1";
  return tag;
}

function renderFaces(state) {

  elements.faceLayer.replaceChildren();
  return;
  if (!state.status.diarization || !state.status.camera_connected) return;

  state.faces.forEach((face) => {
    const speakerId = Number(face.speaker_id);
    const box = document.createElement("div");

    box.className = `face-box${face.active ? " active" : ""}${speakerId ? "" : " unknown"}`;
    box.style.setProperty("--speaker-color", SPEAKER_COLORS[speakerId] || "#667085");
    box.style.left = `${face.x}%`;
    box.style.top = `${face.y}%`;
    box.style.width = `${face.width}%`;
    box.style.height = `${face.height}%`;

    if (speakerId) {
      const label = document.createElement("span");
      label.className = "face-label";
      label.textContent = speakerName(state.speakers, speakerId);
      box.appendChild(label);
    }

    if (face.active) {
      const badge = document.createElement("span");
      badge.className = "speaking-badge";
      badge.textContent = "Speaking";
      box.appendChild(badge);
    }
    elements.faceLayer.appendChild(box);
  });
}

function renderSpeakerCards(state) {
  document.querySelectorAll(".speaker-card").forEach((card) => {
    const speakerId = Number(card.dataset.speakerId);
    const active = state.status.recording && speakerId === Number(state.active_speaker_id);
    card.classList.toggle("active", active);
  });
}

function renderCaptions(state) {
  const fragment = document.createDocumentFragment();
  const captions = state.captions.slice(-7);

  captions.forEach((caption) => {
    const row = document.createElement("article");
    row.className = "caption-row";
    row.appendChild(createSpeakerSlot(state.speakers, Number(caption.speaker_id)));

    const text = document.createElement("p");
    text.textContent = caption.text;
    row.appendChild(text);

    const timestamp = document.createElement("time");
    timestamp.textContent = caption.time;
    row.appendChild(timestamp);
    fragment.appendChild(row);
  });

  if (state.partial?.text) {
    const speakerId = Number(state.partial.speaker_id);
    const row = document.createElement("article");
    row.className = "caption-row partial";
    row.style.setProperty("--active-color", SPEAKER_COLORS[speakerId] || "#667085");
    row.appendChild(createSpeakerSlot(state.speakers, speakerId));

    const text = document.createElement("p");
    text.textContent = state.partial.text;
    row.appendChild(text);

    const dots = document.createElement("span");
    dots.className = "typing-dots";
    dots.setAttribute("aria-label", "Recognizing speech");
    dots.innerHTML = "<i></i><i></i><i></i>";
    row.appendChild(dots);
    fragment.appendChild(row);
  }

  if (!captions.length && !state.partial?.text) {
    const empty = document.createElement("div");
    empty.className = "empty-caption";
    empty.textContent = "Waiting for captions.";
    fragment.appendChild(empty);
  }

  elements.captionList.replaceChildren(fragment);
}

function renderCamera(status) {
  const connected = Boolean(status.camera_connected);
  if (connected && !cameraStreamStarted) {
    elements.cameraStream.src = "/camera.mjpg";
    cameraStreamStarted = true;
  }
  elements.cameraStream.classList.toggle("is-hidden", !connected);
  elements.cameraPlaceholder.hidden = connected;
  elements.cameraMessage.textContent = status.error ? "Check the camera connection." : "Waiting for the camera.";
  elements.cameraLight.classList.toggle("live", connected);
  elements.cameraDot.classList.toggle("live", connected);
  elements.cameraStatus.textContent = connected ? "Camera connected" : "Waiting for the camera";
}

function renderControls(status) {

  const model = status.model !== false;
  elements.modelToggle.classList.toggle("on", model);
  elements.modelButton.setAttribute("aria-pressed", String(model));

  const sttRaw = status.stt_raw !== false;
  elements.sttLabel.textContent = sttRaw ? "Microphones" : "Separated voices";
  elements.sttButton.setAttribute("aria-pressed", String(sttRaw));
}

function renderRuntime(state) {
  const { status } = state;
  const activeSpeakerId = Number(state.active_speaker_id || state.partial?.speaker_id || 0);
  const listening = Boolean(status.recording && (status.audio_connected || status.demo));
  elements.recognitionIndicator.className = `status-indicator ${listening ? "live" : "idle"}`;
  elements.recognitionLabel.textContent = listening
    ? activeSpeakerId
      ? `${speakerName(state.speakers, activeSpeakerId)}: recognizing speech`
      : "Recognizing speech"
    : "Waiting for speech";

  const microphoneConnected = Boolean(status.audio_connected);
  elements.microphoneDot.classList.toggle("live", microphoneConnected);
  elements.microphoneStatus.textContent = microphoneConnected ? "4-channel microphones connected" : "Waiting for the 4-channel microphones";
  const ready = Boolean(status.demo || (status.doa_ready && status.stt_ready));
  elements.runtimeDot.classList.toggle("live", ready);
  elements.runtimeStatus.textContent = ready ? "DOA, speaker matching and STT running" : "Starting DOA, speaker matching and STT";

  elements.errorToast.hidden = !status.error;
  elements.errorToast.textContent = status.error || "";
}

function renderState(state) {
  renderControls(state.status);
  renderCamera(state.status);
  renderFaces(state);
  renderSpeakerCards(state);
  renderCaptions(state);
  renderRuntime(state);
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request failed: ${response.status}`);
    renderState(await response.json());
  } catch (error) {
    elements.runtimeDot.classList.remove("live");
    elements.runtimeStatus.textContent = "Reconnecting to the UI server";
  }
}

async function pollState() {
  await refreshState();
  pollTimer = window.setTimeout(pollState, POLL_INTERVAL_MS);
}

async function toggleControl(action) {
  try {
    const response = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!response.ok) throw new Error(`Control request failed: ${response.status}`);
    await refreshState();
  } catch (error) {
    elements.errorToast.hidden = false;
    elements.errorToast.textContent = error.message;
  }
}

elements.modelButton.addEventListener("click", () => toggleControl("model"));
elements.sttButton.addEventListener("click", () => toggleControl("stt_raw"));
elements.resetButton.addEventListener("click", () => toggleControl("reset_captions"));
elements.settingsButton.addEventListener("click", () => {
  const willOpen = elements.settingsPopover.hidden;
  elements.settingsPopover.hidden = !willOpen;
  elements.settingsButton.setAttribute("aria-expanded", String(willOpen));
});
elements.settingsClose.addEventListener("click", () => {
  elements.settingsPopover.hidden = true;
  elements.settingsButton.setAttribute("aria-expanded", "false");
});
elements.largeTextInput.addEventListener("change", () => {
  elements.screen.classList.toggle("large-text", elements.largeTextInput.checked);
});
window.addEventListener("resize", fitScreen);
window.addEventListener("beforeunload", () => window.clearTimeout(pollTimer));

fitScreen();
initializeWaveforms();
pollState();
