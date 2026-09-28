// Leave empty when Flask serves the frontend and API from the same origin.
const API_BASE = "";
const REQUEST_ERROR = "Unable to analyze the text. Please make sure the server is running and try again.";

const analyzerForm = document.getElementById("analyzerForm");
const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnLabel = analyzeBtn.querySelector(".btn-label");
const exampleChips = document.querySelectorAll(".example-chip");
const errorMsg = document.getElementById("errorMsg");
const result = document.getElementById("result");
const resultEmoji = document.getElementById("resultEmoji");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceTrack = document.getElementById("confidenceTrack");
const tokensSection = document.getElementById("tokensSection");
const tokensList = document.getElementById("tokensList");
const resultNote = document.getElementById("resultNote");
const modelStatus = document.getElementById("modelStatus");
const modelStatusText = document.getElementById("modelStatusText");

const LABEL_CLASS = { Negative: "neg", Neutral: "neu", Positive: "pos" };
const LABEL_EMOJI = { Negative: "😞", Neutral: "😐", Positive: "😄" };
const probabilityElements = {
  Negative: { fill: document.getElementById("barNeg"), value: document.getElementById("pctNeg"), track: document.getElementById("barNeg").parentElement },
  Neutral: { fill: document.getElementById("barNeu"), value: document.getElementById("pctNeu"), track: document.getElementById("barNeu").parentElement },
  Positive: { fill: document.getElementById("barPos"), value: document.getElementById("pctPos"), track: document.getElementById("barPos").parentElement },
};

textInput.addEventListener("input", updateCharacterCount);
clearBtn.addEventListener("click", clearInput);
analyzerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  analyze();
});

exampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    textInput.value = chip.dataset.text || "";
    updateCharacterCount();
    textInput.focus();
    analyze();
  });
});

textInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    analyze();
  }
});

function updateCharacterCount() {
  charCount.textContent = textInput.value.length;
}

function clearInput() {
  textInput.value = "";
  updateCharacterCount();
  hideError();
  result.hidden = true;
  textInput.focus();
}

async function analyze() {
  const text = textInput.value.trim();
  hideError();

  if (!text) {
    result.hidden = true;
    showError("Please enter some text to analyze.");
    textInput.focus();
    return;
  }

  setLoading(true);
  result.hidden = true;
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      showError(REQUEST_ERROR);
      return;
    }

    const data = await response.json();
    renderResult(data);
  } catch {
    showError(REQUEST_ERROR);
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  const label = LABEL_CLASS[data.label] ? data.label : "Neutral";
  const confidence = toPercentage(data.confidence);
  const probabilities = data.probabilities || {};

  result.dataset.sentiment = label;
  resultEmoji.textContent = data.emoji || LABEL_EMOJI[label];
  resultLabel.textContent = label;
  resultLabel.className = `result-label ${LABEL_CLASS[label]}`;
  resultConfidence.textContent = formatPercentage(confidence);
  confidenceValue.textContent = `${formatPercentage(confidence)}%`;
  confidenceTrack.setAttribute("aria-valuenow", String(confidence));
  confidenceFill.style.width = "0%";

  const probabilityValues = {};
  Object.entries(probabilityElements).forEach(([name, elements]) => {
    const percentage = toPercentage(probabilities[name]);
    elements.value.textContent = `${formatPercentage(percentage)}%`;
    elements.track.setAttribute("aria-valuenow", String(percentage));
    elements.fill.style.width = "0%";
    probabilityValues[name] = percentage;
  });

  tokensList.replaceChildren();
  const tokens = Array.isArray(data.tokens) ? data.tokens : [];
  tokens.forEach((token) => {
    const badge = document.createElement("span");
    badge.textContent = token;
    tokensList.appendChild(badge);
  });
  tokensSection.hidden = tokens.length === 0;

  if (data.warning) {
    resultNote.textContent = data.warning;
    resultNote.hidden = false;
  } else {
    resultNote.textContent = "";
    resultNote.hidden = true;
  }

  result.hidden = false;
  requestAnimationFrame(() => {
    confidenceFill.style.width = `${confidence}%`;
    Object.entries(probabilityElements).forEach(([name, elements]) => {
      elements.fill.style.width = `${probabilityValues[name]}%`;
    });
  });
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function toPercentage(value) {
  const percentage = Number(value);
  if (!Number.isFinite(percentage)) return 0;
  return Math.min(100, Math.max(0, percentage));
}

function formatPercentage(value) {
  return value.toFixed(2);
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.classList.toggle("loading", isLoading);
  analyzeBtn.setAttribute("aria-busy", String(isLoading));
  btnLabel.textContent = isLoading ? "Analyzing..." : "Analyze Sentiment";
  textInput.disabled = isLoading;
  clearBtn.disabled = isLoading;
  exampleChips.forEach((chip) => { chip.disabled = isLoading; });
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

async function loadModelStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) {
      setModelUnavailable();
      return;
    }

    const data = await response.json();
    if (data.status !== "ok") {
      setModelUnavailable();
      return;
    }

    modelStatus.classList.add("is-ready");
    modelStatus.classList.remove("is-unavailable");
    modelStatusText.textContent = "Model Ready";

    const metrics = data.metrics || {};
    if (metrics.accuracy != null && Number.isFinite(Number(metrics.accuracy))) {
      document.getElementById("statAcc").textContent = `${Math.round(Number(metrics.accuracy) * 100)}%`;
    }
    if (metrics.f1 != null && Number.isFinite(Number(metrics.f1))) {
      document.getElementById("statF1").textContent = Number(metrics.f1).toFixed(2);
    }
    if (data.vocab_size != null) {
      document.getElementById("statVocab").textContent = Number(data.vocab_size).toLocaleString();
    }
  } catch {
    setModelUnavailable();
  }
}

function setModelUnavailable() {
  modelStatus.classList.remove("is-ready");
  modelStatus.classList.add("is-unavailable");
  modelStatusText.textContent = "Model unavailable";
}

loadModelStatus();
