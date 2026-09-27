// If you deploy the frontend separately from the backend (e.g. static
// hosting + a separately hosted API), set this to the backend's full URL,
// e.g. "https://your-backend.onrender.com". Leave empty to call the same
// origin the page is served from (the default when Flask serves both).
const API_BASE = "";

const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const result = document.getElementById("result");

const resultEmoji = document.getElementById("resultEmoji");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");

const segNeg = document.getElementById("segNeg");
const segNeu = document.getElementById("segNeu");
const segPos = document.getElementById("segPos");
const pctNeg = document.getElementById("pctNeg");
const pctNeu = document.getElementById("pctNeu");
const pctPos = document.getElementById("pctPos");
const tokensList = document.getElementById("tokensList");

const LABEL_CLASS = { Negative: "neg", Neutral: "neu", Positive: "pos" };

textInput.addEventListener("input", () => {
  charCount.textContent = textInput.value.length;
});

document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textInput.value = chip.dataset.text;
    charCount.textContent = textInput.value.length;
    textInput.focus();
    analyze();
  });
});

textInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    analyze();
  }
});

analyzeBtn.addEventListener("click", analyze);

async function analyze() {
  const text = textInput.value.trim();
  hideError();

  if (!text) {
    showError("Type something first — a sentence or two is enough.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong. Please try again.");
      result.hidden = true;
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Couldn't reach the backend. Is the Flask server running?");
    result.hidden = true;
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  result.hidden = false;

  resultEmoji.textContent = data.emoji;
  resultLabel.textContent = data.label;
  resultLabel.className = "result-label " + (LABEL_CLASS[data.label] || "");
  resultConfidence.textContent = data.confidence;

  const p = data.probabilities || {};
  const neg = p.Negative ?? 0;
  const neu = p.Neutral ?? 0;
  const pos = p.Positive ?? 0;

  segNeg.style.width = neg + "%";
  segNeu.style.width = neu + "%";
  segPos.style.width = pos + "%";
  pctNeg.textContent = neg + "%";
  pctNeu.textContent = neu + "%";
  pctPos.textContent = pos + "%";

  tokensList.innerHTML = "";
  (data.tokens || []).forEach((tok) => {
    const span = document.createElement("span");
    span.textContent = tok;
    tokensList.appendChild(span);
  });

  if (data.warning) {
    showError(data.warning);
  }

  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.classList.toggle("loading", isLoading);
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

// Load model stats for the intro column
async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return;
    const data = await res.json();
    const m = data.metrics || {};
    if (m.accuracy != null) {
      document.getElementById("statAcc").textContent =
        Math.round(m.accuracy * 100) + "%";
    }
    if (m.f1 != null) {
      document.getElementById("statF1").textContent = m.f1.toFixed(2);
    }
    if (data.vocab_size != null) {
      document.getElementById("statVocab").textContent = data.vocab_size;
    }
  } catch (err) {
    // Silently ignore — stats are a nice-to-have, not critical.
  }
}

loadStats();
