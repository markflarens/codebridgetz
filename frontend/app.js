(() => {
  const API_BASE = "";  // same-origin; change if backend is hosted separately

  const screens = {};
  document.querySelectorAll("[data-screen]").forEach(el => { screens[el.id] = el; });

  function showScreen(id) {
    Object.values(screens).forEach(el => el.hidden = true);
    screens[id].hidden = false;
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  document.querySelectorAll("[data-back-to]").forEach(btn => {
    btn.addEventListener("click", () => {
      resetCapture();
      showScreen(btn.dataset.backTo);
    });
  });

  // ---------------- Screen 1 -> 2 ----------------
  document.getElementById("btn-start").addEventListener("click", () => {
    showScreen("screen-capture");
  });

  // ---------------- Screen 2: capture ----------------
  const inputCamera = document.getElementById("input-camera");
  const inputGallery = document.getElementById("input-gallery");
  const previewImg = document.getElementById("preview-img");
  const placeholder = document.getElementById("capture-placeholder");
  const btnAnalyze = document.getElementById("btn-analyze");
  let selectedFile = null;

  document.getElementById("btn-camera").addEventListener("click", () => inputCamera.click());
  document.getElementById("btn-gallery").addEventListener("click", () => inputGallery.click());

  [inputCamera, inputGallery].forEach(input => {
    input.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      selectedFile = file;
      const url = URL.createObjectURL(file);
      previewImg.src = url;
      previewImg.hidden = false;
      placeholder.hidden = true;
      btnAnalyze.disabled = false;
    });
  });

  function resetCapture() {
    selectedFile = null;
    previewImg.src = "";
    previewImg.hidden = true;
    placeholder.hidden = false;
    btnAnalyze.disabled = true;
    inputCamera.value = "";
    inputGallery.value = "";
  }

  // ---------------- Screen 3: processing ----------------
  const scanningImg = document.getElementById("scanning-img");
  const scannerStatus = document.getElementById("scanner-status");
  const STATUS_MESSAGES = [
    "Locating calibration marker…",
    "Correcting perspective…",
    "Scanning inner boundary…",
    "Cross-checking detection methods…",
  ];

  function runStatusCycle() {
    let i = 0;
    scannerStatus.textContent = STATUS_MESSAGES[0];
    return setInterval(() => {
      i = (i + 1) % STATUS_MESSAGES.length;
      scannerStatus.textContent = STATUS_MESSAGES[i];
    }, 1100);
  }

  btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;
    scanningImg.src = previewImg.src;
    showScreen("screen-processing");
    const statusTimer = runStatusCycle();

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/api/measure`, { method: "POST", body: form });
      const data = await res.json();
      clearInterval(statusTimer);
      if (data.status === "success") {
        renderSuccess(data);
        showScreen("screen-result");
      } else {
        renderRetake(data);
        showScreen("screen-retake");
      }
    } catch (err) {
      clearInterval(statusTimer);
      renderRetake({
        reason: "NETWORK_ERROR",
        message: "Could not reach the measurement service. Check your connection and try again.",
      });
      showScreen("screen-retake");
    }
  });

  // ---------------- Screen 4: success ----------------
  function renderSuccess(data) {
    document.getElementById("result-overlay-img").src = data.overlay_image || "";
    document.getElementById("reading-diameter").textContent = data.diameter_mm.toFixed(2);
    document.getElementById("reading-size").textContent = data.ring_size;

    const flag = document.getElementById("reading-size-flag");
    flag.textContent = data.ring_size_in_range ? "" : "outside standard US finger-ring range";

    document.getElementById("data-spread").textContent = `± ${data.detection_spread_mm.toFixed(2)} mm`;

    const methods = Object.keys(data.family_estimates || {});
    document.getElementById("data-methods").textContent = `${methods.length} (${methods.join(", ")})`;

    document.getElementById("data-time").textContent = `${(data.processing_time_ms / 1000).toFixed(1)} s`;
    document.getElementById("data-standard").textContent = data.sizing_standard;
    document.getElementById("data-source").textContent = `Source: ${data.sizing_source}`;
    document.getElementById("data-rounding").textContent = data.rounding_rule;
  }

  document.getElementById("btn-retry-success").addEventListener("click", () => {
    resetCapture();
    showScreen("screen-capture");
  });

  // ---------------- Screen 5: retake ----------------
  function renderRetake(data) {
    document.getElementById("retake-code").textContent = data.reason || "UNKNOWN";
    document.getElementById("retake-message").textContent = data.message || "Please retake the photo.";

    const wrap = document.getElementById("retake-overlay-wrap");
    const img = document.getElementById("retake-overlay-img");
    if (data.overlay_image) {
      img.src = data.overlay_image;
      wrap.hidden = false;
    } else {
      wrap.hidden = true;
    }
  }

  document.getElementById("btn-retry-retake").addEventListener("click", () => {
    resetCapture();
    showScreen("screen-capture");
  });

  showScreen("screen-instructions");
})();
