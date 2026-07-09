/* ============================================================
   AuraSentiment — Premium SaaS AI Client Script
   Preserves all original fetch() API integration logic.
   ============================================================ */

/* ───────────────────────────────────────────
   CONFIG
─────────────────────────────────────────── */
const API_BASE_URL = "https://sentiment-analysis-web-app-7odn.onrender.com";

/* ───────────────────────────────────────────
   SPLASH SCREEN + PARTICLE CANVAS
─────────────────────────────────────────── */
(function initSplash() {
  const splash   = document.getElementById("splash-screen");
  const mainApp  = document.getElementById("main-app");
  const loaderFill  = document.getElementById("loader-fill");
  const loaderLabel = document.getElementById("loader-label");
  const canvas   = document.getElementById("particle-canvas");
  const ctx      = canvas.getContext("2d");

  // — Particle System —
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  window.addEventListener("resize", () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  });

  const PARTICLE_COUNT = 70;
  const particles = [];

  function randomBetween(a, b) { return a + Math.random() * (b - a); }

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: randomBetween(0, canvas.width),
      y: randomBetween(0, canvas.height),
      r: randomBetween(1, 3.5),
      dx: randomBetween(-0.3, 0.3),
      dy: randomBetween(-0.5, -0.1),
      alpha: randomBetween(0.1, 0.7),
      hue: randomBetween(240, 290),
    });
  }

  function drawParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 90%, 70%, ${p.alpha})`;
      ctx.shadowBlur = 8;
      ctx.shadowColor = `hsla(${p.hue}, 90%, 70%, 0.5)`;
      ctx.fill();

      p.x += p.dx;
      p.y += p.dy;
      if (p.y < -10) { p.y = canvas.height + 10; p.x = randomBetween(0, canvas.width); }
      if (p.x < -10 || p.x > canvas.width + 10) p.x = randomBetween(0, canvas.width);
    });
  }

  let animFrame;
  function loop() { drawParticles(); animFrame = requestAnimationFrame(loop); }
  loop();

  // — Loader Progress —
  const steps = [
    { pct: 15,  label: "Loading ML model..." },
    { pct: 40,  label: "Preparing TF-IDF vectorizer..." },
    { pct: 65,  label: "Setting up FastAPI bridge..." },
    { pct: 85,  label: "Calibrating sentiment engine..." },
    { pct: 100, label: "Ready!" },
  ];
  let stepIndex = 0;
  function advanceLoader() {
    if (stepIndex >= steps.length) return;
    const s = steps[stepIndex++];
    loaderFill.style.width  = s.pct + "%";
    loaderLabel.textContent = s.label;
    if (stepIndex < steps.length) {
      setTimeout(advanceLoader, randomBetween(350, 550));
    } else {
      // Finished — fade out splash
      setTimeout(() => {
        splash.classList.add("fade-out");
        setTimeout(() => {
          splash.style.display = "none";
          mainApp.classList.remove("hidden");
          cancelAnimationFrame(animFrame);
          initApp();
        }, 700);
      }, 400);
    }
  }
  setTimeout(advanceLoader, 200);
})();

/* ───────────────────────────────────────────
   MAIN APP INIT
─────────────────────────────────────────── */
function initApp() {
  initNavbar();
  initHamburger();
  initScrollReveal();
  initAnalyzer();
}

/* ───────────────────────────────────────────
   NAVBAR — scroll effect
─────────────────────────────────────────── */
function initNavbar() {
  const navbar = document.getElementById("navbar");
  function onScroll() {
    navbar.classList.toggle("scrolled", window.scrollY > 30);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

/* ───────────────────────────────────────────
   HAMBURGER MENU
─────────────────────────────────────────── */
function initHamburger() {
  const btn   = document.getElementById("hamburger");
  const menu  = document.getElementById("mobile-menu");
  const links = menu.querySelectorAll(".mobile-link");

  btn.addEventListener("click", () => {
    btn.classList.toggle("open");
    menu.classList.toggle("open");
  });
  links.forEach(l => l.addEventListener("click", () => {
    btn.classList.remove("open");
    menu.classList.remove("open");
  }));
}

/* ───────────────────────────────────────────
   SCROLL REVEAL (IntersectionObserver)
─────────────────────────────────────────── */
function initScrollReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!els.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        // Staggered delay for sibling cards
        const siblings = [...e.target.parentElement.querySelectorAll(".reveal")];
        const idx = siblings.indexOf(e.target);
        e.target.style.transitionDelay = `${idx * 80}ms`;
        e.target.classList.add("visible");
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  els.forEach(el => observer.observe(el));
}

/* ───────────────────────────────────────────
   TOAST NOTIFICATION
─────────────────────────────────────────── */
let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById("toast");
  const txt   = document.getElementById("toast-text");
  txt.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3500);
}

/* ───────────────────────────────────────────
   ANALYZER — preserved fetch() + enhanced UI
─────────────────────────────────────────── */
function initAnalyzer() {
  // DOM references (same IDs as original)
  const textInput  = document.getElementById("text-input");
  const charCount  = document.getElementById("char-count");
  const predictBtn = document.getElementById("predict-btn");
  const btnText    = document.getElementById("btn-text");
  const btnIcon    = document.getElementById("btn-icon");
  const btnSpinner = document.getElementById("btn-spinner");
  const resultCard = document.getElementById("result-card");
  const textPreview= document.getElementById("text-preview");
  const sentimentBadge = document.getElementById("sentiment-badge");
  const sentimentIcon  = document.getElementById("sentiment-icon");
  const sentimentLabel = document.getElementById("sentiment-label");
  const confidenceContainer = document.getElementById("confidence-container");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceBar   = document.getElementById("confidence-bar");
  const resetBtn        = document.getElementById("reset-btn");

  // ── Character Counter ──
  textInput.addEventListener("input", () => {
    const len = textInput.value.length;
    charCount.textContent = len;
    charCount.style.color = len >= 450
      ? "var(--neg)"
      : "var(--txt-3)";
  });

  // ── Reset button ──
  resetBtn.addEventListener("click", () => {
    resultCard.classList.add("hidden");
    resultCard.className = "result-card hidden";
    textInput.value = "";
    charCount.textContent = "0";
    textInput.focus();
  });

  // ── Primary Predict Handler ──
  predictBtn.addEventListener("click", async () => {
    const rawText = textInput.value;

    // Validation — no browser alert, use toast
    if (!rawText || !rawText.trim()) {
      showToast("Please enter some text before analyzing.");
      textInput.focus();
      return;
    }

    hideResults();
    setLoadingState(true);

    try {
      // ── Fetch API (original logic preserved) ──
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawText }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Server returned an error status.");
      }

      displayPrediction(data);

    } catch (error) {
      console.error("Sentiment classification failure:", error);
      showToast(
        error.message ||
        "Cannot reach the API. Make sure the backend server is running."
      );
    } finally {
      setLoadingState(false);
    }
  });

  // ── Also trigger on Ctrl/Cmd + Enter ──
  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      predictBtn.click();
    }
  });

  /* ── UI Helpers ── */
  function setLoadingState(loading) {
    if (loading) {
      predictBtn.disabled = true;
      btnText.textContent = "Analyzing...";
      btnIcon.classList.add("hidden");
      btnSpinner.classList.remove("hidden");
    } else {
      predictBtn.disabled = false;
      btnText.textContent = "Analyze Sentiment";
      btnIcon.classList.remove("hidden");
      btnSpinner.classList.add("hidden");
    }
  }

  function hideResults() {
    resultCard.classList.add("hidden");
  }

  /* ── Display Prediction (enhanced) ── */
  function displayPrediction(data) {
    const sentiment = data.prediction.toLowerCase().trim(); // positive | neutral | negative
    const text      = data.text;
    const confidence = data.confidence; // e.g. "94.82%"

    // Text preview
    textPreview.textContent = text;

    // Reset classes
    sentimentBadge.className = "sentiment-badge";
    confidenceBar.className  = "conf-fill";
    resultCard.className     = "result-card";

    // Emoji, icon, badge, glow
    let emoji = "😊", iconClass = "fa-face-smile";

    if (sentiment === "positive") {
      emoji = "😊"; iconClass = "fa-face-smile";
      sentimentBadge.classList.add("positive");
      confidenceBar.classList.add("positive");
      resultCard.classList.add("glow-pos");
      sentimentLabel.textContent = "Positive";
      confidenceValue.style.color = "var(--pos)";
    } else if (sentiment === "neutral") {
      emoji = "😐"; iconClass = "fa-face-meh";
      sentimentBadge.classList.add("neutral");
      confidenceBar.classList.add("neutral");
      resultCard.classList.add("glow-neu");
      sentimentLabel.textContent = "Neutral";
      confidenceValue.style.color = "var(--neu)";
    } else {
      emoji = "😞"; iconClass = "fa-face-frown";
      sentimentBadge.classList.add("negative");
      confidenceBar.classList.add("negative");
      resultCard.classList.add("glow-neg");
      sentimentLabel.textContent = "Negative";
      confidenceValue.style.color = "var(--neg)";
    }

    document.getElementById("result-emoji").textContent = emoji;
    sentimentIcon.className = `fa-solid ${iconClass}`;

    // Confidence bar (animated)
    if (confidence) {
      confidenceValue.textContent = confidence;
      confidenceContainer.classList.remove("hidden");
      // Animate bar after short delay for CSS transition
      setTimeout(() => { confidenceBar.style.width = confidence; }, 80);
    } else {
      confidenceContainer.classList.add("hidden");
    }

    // Show card with animation
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}
