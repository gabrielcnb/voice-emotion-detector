/**
 * Voice Emotion Detector - Frontend
 * Mic recording (MediaRecorder API) + drag-and-drop upload
 */

const EMOTION_EMOJIS = {
    neutral: "\uD83D\uDE10",
    happy: "\uD83D\uDE0A",
    sad: "\uD83D\uDE22",
    angry: "\uD83D\uDE20",
    fearful: "\uD83D\uDE28",
    disgusted: "\uD83E\uDD22",
    surprised: "\uD83D\uDE32",
};

// Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const recordBtn = document.getElementById("recordBtn");
const recordTimer = document.getElementById("recordTimer");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorDiv = document.getElementById("error");
const errorText = document.getElementById("errorText");
const resultEmoji = document.getElementById("resultEmoji");
const resultEmotion = document.getElementById("resultEmotion");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceText = document.getElementById("confidenceText");
const probChart = document.getElementById("probChart");

// State
let mediaRecorder = null;
let audioChunks = [];
let timerInterval = null;
let recordStart = 0;

// --- Drag & Drop ---
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("audio/")) {
        uploadFile(file);
    }
});

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
    fileInput.value = "";
});

// --- Upload ---
async function uploadFile(file) {
    showLoading();
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/predict/upload", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        if (res.ok) {
            showResults(data);
        } else {
            showError(data.error || "Erro desconhecido");
        }
    } catch (err) {
        showError("Erro de conexão: " + err.message);
    }
}

// --- Recording ---
recordBtn.addEventListener("click", async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
    } else {
        startRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            clearInterval(timerInterval);

            const blob = new Blob(audioChunks, { type: "audio/webm" });
            await uploadRecording(blob);
        };

        mediaRecorder.start();
        recordBtn.classList.add("recording");
        recordBtn.querySelector("span:last-child") ||
            (recordBtn.textContent = "");
        recordBtn.innerHTML = '<span class="record-dot"></span> Parar';
        recordStart = Date.now();
        updateTimer();
        timerInterval = setInterval(updateTimer, 100);
    } catch (err) {
        showError("Não foi possível acessar o microfone: " + err.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.classList.remove("recording");
        recordBtn.innerHTML = '<span class="record-dot"></span> Gravar';
    }
}

function updateTimer() {
    const elapsed = (Date.now() - recordStart) / 1000;
    const min = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const sec = Math.floor(elapsed % 60).toString().padStart(2, "0");
    recordTimer.textContent = `${min}:${sec}`;
}

async function uploadRecording(blob) {
    showLoading();
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    try {
        const res = await fetch("/api/predict/record", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        if (res.ok) {
            showResults(data);
        } else {
            showError(data.error || "Erro desconhecido");
        }
    } catch (err) {
        showError("Erro de conexão: " + err.message);
    }
}

// --- UI Updates ---
function showLoading() {
    loading.classList.remove("hidden");
    results.classList.add("hidden");
    errorDiv.classList.add("hidden");
}

function showError(msg) {
    loading.classList.add("hidden");
    results.classList.add("hidden");
    errorDiv.classList.remove("hidden");
    errorText.textContent = msg;
}

function showResults(data) {
    loading.classList.add("hidden");
    errorDiv.classList.add("hidden");
    results.classList.remove("hidden");

    // Main result
    resultEmoji.textContent = data.emoji || EMOTION_EMOJIS[data.emotion] || "";
    resultEmotion.textContent = translateEmotion(data.emotion);
    confidenceBar.style.width = (data.confidence * 100) + "%";
    confidenceText.textContent = `Confiança: ${(data.confidence * 100).toFixed(1)}%`;

    // Probability bars
    probChart.innerHTML = "";
    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    const maxProb = sorted[0][1];

    for (const [emotion, prob] of sorted) {
        const row = document.createElement("div");
        row.className = "prob-row";

        const emoji = document.createElement("span");
        emoji.className = "prob-emoji";
        emoji.textContent = EMOTION_EMOJIS[emotion] || "";

        const label = document.createElement("span");
        label.className = "prob-label";
        label.textContent = translateEmotion(emotion);

        const barContainer = document.createElement("div");
        barContainer.className = "prob-bar-container";

        const bar = document.createElement("div");
        bar.className = "prob-bar";
        bar.dataset.emotion = emotion;
        bar.style.width = (prob / maxProb * 100) + "%";

        barContainer.appendChild(bar);

        const value = document.createElement("span");
        value.className = "prob-value";
        value.textContent = (prob * 100).toFixed(1) + "%";

        row.appendChild(emoji);
        row.appendChild(label);
        row.appendChild(barContainer);
        row.appendChild(value);
        probChart.appendChild(row);
    }

    // Scroll to results
    results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function translateEmotion(emotion) {
    const translations = {
        neutral: "Neutro",
        happy: "Feliz",
        sad: "Triste",
        angry: "Com raiva",
        fearful: "Com medo",
        disgusted: "Enojado",
        surprised: "Surpreso",
    };
    return translations[emotion] || emotion;
}
