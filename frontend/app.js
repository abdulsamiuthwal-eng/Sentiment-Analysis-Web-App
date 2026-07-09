/* ==========================================================================
   AuraSentiment Client Integration Logic
   ========================================================================== */

const API_BASE_URL = "http://127.0.0.1:8000";

// DOM Elements Selection
const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const predictBtn = document.getElementById("predict-btn");
const btnText = document.getElementById("btn-text");
const btnIcon = document.getElementById("btn-icon");
const btnSpinner = document.getElementById("btn-spinner");

const errorMessage = document.getElementById("error-message");
const errorText = document.getElementById("error-text");

const resultCard = document.getElementById("result-card");
const textPreview = document.getElementById("text-preview");
const sentimentBadge = document.getElementById("sentiment-badge");
const sentimentIcon = document.getElementById("sentiment-icon");
const sentimentLabel = document.getElementById("sentiment-label");
const confidenceContainer = document.getElementById("confidence-container");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBar = document.getElementById("confidence-bar");

// Real-time Character Counter Event Listener
textInput.addEventListener("input", () => {
    const len = textInput.value.length;
    charCount.textContent = len;
    
    // Visual indicator when approaching max length
    if (len >= 450) {
        charCount.style.color = "var(--color-neg)";
    } else {
        charCount.style.color = "hsla(215, 20%, 65%, 0.5)";
    }
});

// Primary Predict Event Handler
predictBtn.addEventListener("click", async () => {
    const rawText = textInput.value;
    
    // 1. Client-side Validation (Empty / Whitespace check)
    if (!rawText || !rawText.trim()) {
        showError("Please enter some text before classifying.");
        hideResults();
        return;
    }
    
    // Clear previous states
    hideError();
    hideResults();
    setLoadingState(true);

    try {
        // 2. Fetch API Request
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: rawText })
        });

        const data = await response.json();

        // 3. Response Validation
        if (!response.ok) {
            throw new Error(data.detail || "Server returned an error status.");
        }

        // 4. Update UI with Results
        displayPrediction(data);

    } catch (error) {
        console.error("Sentiment classification failure:", error);
        showError(error.message || "Failed to communicate with the classification API. Make sure the backend server is running.");
    } finally {
        setLoadingState(false);
    }
});

// UI State Modifiers
function setLoadingState(isLoading) {
    if (isLoading) {
        predictBtn.disabled = true;
        btnText.textContent = "Analyzing Text...";
        btnIcon.classList.add("hidden");
        btnSpinner.classList.remove("hidden");
    } else {
        predictBtn.disabled = false;
        btnText.textContent = "Classify Sentiment";
        btnIcon.classList.remove("hidden");
        btnSpinner.classList.add("hidden");
    }
}

function showError(msg) {
    errorText.textContent = msg;
    errorMessage.classList.remove("hidden");
}

function hideError() {
    errorMessage.classList.add("hidden");
}

function hideResults() {
    resultCard.classList.add("hidden");
}

function displayPrediction(data) {
    const sentiment = data.prediction.toLowerCase().trim(); // expected: positive, neutral, negative
    const text = data.text;
    const confidence = data.confidence; // e.g. "94.82%"

    // Set preview text
    textPreview.textContent = `"${text}"`;

    // Configure sentiment-specific icons and classes
    sentimentBadge.className = "sentiment-badge"; // Reset class list
    confidenceBar.className = "progress-bar"; // Reset class list
    
    let iconClass = "";
    if (sentiment === "positive") {
        sentimentBadge.classList.add("positive");
        confidenceBar.classList.add("positive");
        iconClass = "fa-face-smile";
        sentimentLabel.textContent = "Positive 😊";
    } else if (sentiment === "neutral") {
        sentimentBadge.classList.add("neutral");
        confidenceBar.classList.add("neutral");
        iconClass = "fa-face-meh";
        sentimentLabel.textContent = "Neutral 😐";
    } else {
        sentimentBadge.classList.add("negative");
        confidenceBar.classList.add("negative");
        iconClass = "fa-face-frown";
        sentimentLabel.textContent = "Negative 😞";
    }
    
    // Set appropriate font-awesome icon class
    sentimentIcon.className = `fa-solid ${iconClass}`;

    // Display confidence score if available
    if (confidence) {
        confidenceValue.textContent = confidence;
        confidenceBar.style.width = confidence;
        confidenceContainer.classList.remove("hidden");
    } else {
        confidenceContainer.classList.add("hidden");
    }

    // Reveal results card
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
