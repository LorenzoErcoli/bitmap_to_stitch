const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download-link");
const multiDownloadEl = document.getElementById("download-multi");
const presetDownloadEl = document.getElementById("download-preset");
const presetInputEl = document.getElementById("preset-file");
const colorDownloadsEl = document.getElementById("color-downloads");
const progressBarEl = document.getElementById("progress-bar");
const progressTextEl = document.getElementById("progress-text");
const convertBtn = document.getElementById("convert");
const previewBtn = document.getElementById("preview");
const previewPanelEl = document.getElementById("preview-panel");
const previewImageEl = document.getElementById("preview-image");
const previewMaskEl = document.getElementById("preview-mask");
const previewPointsEl = document.getElementById("preview-points");
const previewStatsEl = document.getElementById("preview-stats");
const previewColorsEl = document.getElementById("preview-colors");
const lightboxEl = document.getElementById("image-lightbox");
const lightboxImageEl = document.getElementById("lightbox-image");
const closeLightboxBtn = document.getElementById("close-lightbox");
let pyodideReady = null;
let colorDownloadUrls = [];
let monoDownloadUrl = null;
let multiDownloadUrl = null;
let presetDownloadUrl = null;
let lastPreviewCacheKey = null;
let liveProgressStop = null;
let selectedPreviewColor = null;

function canUseLocalBackend() {
  return (
    window.location.protocol === "http:" &&
    ["127.0.0.1", "localhost"].includes(window.location.hostname)
  );
}

function setProgress(value, label = "") {
  const pct = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  if (progressBarEl) {
    progressBarEl.style.width = `${pct}%`;
  }
  if (progressTextEl) {
    progressTextEl.textContent = label
      ? `Avanzamento: ${pct}% - ${label}`
      : `Avanzamento: ${pct}%`;
  }
}

async function initPyodide() {
  if (!pyodideReady) {
    setProgress(5, "Carico runtime");
    statusEl.textContent = "Carico runtime Python...";
    pyodideReady = (async () => {
      const pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
      });
      await pyodide.loadPackage(["numpy", "pillow"]);
      setProgress(12, "Runtime pronto");
      const appSource = await fetch("app.py").then((res) => res.text());
      await pyodide.runPythonAsync(appSource);
      setProgress(18, "Pipeline caricata");
      return pyodide;
    })();
  }
  return pyodideReady;
}

function readOptions() {
  const val = (id) => document.getElementById(id).value.trim();
  const num = (id, def = 0, parser = parseFloat) => {
    const raw = val(id);
    if (!raw) return def;
    const parsed = parser(raw);
    return Number.isNaN(parsed) ? def : parsed;
  };

  return {
    style: document.getElementById("style").value,
    max_width: num("max-width", 0, parseInt),
    threshold: num("threshold", 200, parseInt),
    sample_colors: val("sample-colors"),
    sample_tolerance: num("sample-tolerance", 0.0),
    exclude_background: document.getElementById("exclude-background").checked,
    background_colors: val("background-colors"),
    background_tolerance: num("background-tolerance", 0.0),
    default_dpi: num("default-dpi", 96.0),
    max_points: num("max-points", 0, parseInt),
    target_density: num("target-density", 0.0),
    scale: num("scale", 1.0),
    analysis_cell_mm: num("analysis-cell-mm", 0.0),
    stroke_width: num("stroke-width", 0.3),
    min_dist: num("min-dist", 1.0),
    reinsertion_rounds: num("reinsertion", 1, parseInt),
    chunk_size: num("chunk-size", 0, parseInt),
    ordering: document.getElementById("ordering").value,
    scanline_band: num("scanline-band", 4, parseInt),
    serpentine: document.getElementById("serpentine").checked,
    grid_cell_size: num("grid-cell", 0, parseInt),
    degrade_drop: num("degrade-drop", 0.3),
    degrade_jitter: num("degrade-jitter", 1.0),
    degrade_seed: val("degrade-seed") || null,
    color_count: num("color-count", 2, parseInt),
  };
}

function setControlValue(id, value) {
  const el = document.getElementById(id);
  if (!el || value === undefined || value === null) {
    return;
  }
  if (el.type === "checkbox") {
    el.checked = Boolean(value);
  } else {
    el.value = String(value);
  }
}

function applyOptions(options) {
  if (!options || typeof options !== "object") {
    throw new Error("Preset non valido: options mancanti.");
  }

  const fieldMap = {
    style: "style",
    max_width: "max-width",
    threshold: "threshold",
    sample_colors: "sample-colors",
    sample_tolerance: "sample-tolerance",
    exclude_background: "exclude-background",
    background_colors: "background-colors",
    background_tolerance: "background-tolerance",
    default_dpi: "default-dpi",
    max_points: "max-points",
    target_density: "target-density",
    scale: "scale",
    analysis_cell_mm: "analysis-cell-mm",
    stroke_width: "stroke-width",
    min_dist: "min-dist",
    reinsertion_rounds: "reinsertion",
    chunk_size: "chunk-size",
    ordering: "ordering",
    scanline_band: "scanline-band",
    serpentine: "serpentine",
    grid_cell_size: "grid-cell",
    degrade_drop: "degrade-drop",
    degrade_jitter: "degrade-jitter",
    degrade_seed: "degrade-seed",
    color_count: "color-count",
  };

  Object.entries(fieldMap).forEach(([key, id]) => {
    setControlValue(id, options[key]);
  });

  document.getElementById("style").dispatchEvent(new Event("change"));
}

function parsePresetText(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    const doc = new DOMParser().parseFromString(text, "image/svg+xml");
    const metadata = doc.querySelector("#bitmap-to-stitch-params");
    if (!metadata || !metadata.textContent) {
      throw new Error("Il file non contiene un preset valido.");
    }
    return JSON.parse(metadata.textContent);
  }
}

function sanitizeFileStem(name) {
  return (name || "stitch")
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9_-]+/gi, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase() || "stitch";
}

function buildPresetPayload(options, sourceFile, result) {
  return {
    schema: "bitmap-to-stitch-preset/v1",
    app: "BitmapToStitch",
    created_at: new Date().toISOString(),
    source_file: sourceFile
      ? {
          name: sourceFile.name,
          size: sourceFile.size,
          type: sourceFile.type || "",
        }
      : null,
    options,
    result_summary: result?.summary || "",
    colors: (result?.colors || []).map((info) => ({
      color: info.color,
      initial_points: info.initial_points,
      final_points: info.final_points,
      discarded_points: info.discarded_points,
      allocated_max_points: info.allocated_max_points,
    })),
  };
}

function escapeXmlText(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function injectSvgMetadata(svgStr, presetPayload, extra = {}) {
  const metadata = {
    ...presetPayload,
    export: {
      ...(presetPayload.export || {}),
      ...extra,
    },
  };
  const metadataEl =
    `<metadata id="bitmap-to-stitch-params" type="application/json">` +
    `${escapeXmlText(JSON.stringify(metadata, null, 2))}` +
    `</metadata>`;
  return String(svgStr || "").replace(/(<svg\b[^>]*>)/, `$1\n${metadataEl}`);
}

function resetPrimaryDownloads() {
  if (downloadEl) {
    downloadEl.style.display = "none";
  }
  if (multiDownloadEl) {
    multiDownloadEl.style.display = "none";
  }
  if (monoDownloadUrl) {
    URL.revokeObjectURL(monoDownloadUrl);
    monoDownloadUrl = null;
  }
  if (multiDownloadUrl) {
    URL.revokeObjectURL(multiDownloadUrl);
    multiDownloadUrl = null;
  }
  if (presetDownloadEl) {
    presetDownloadEl.style.display = "none";
  }
  if (presetDownloadUrl) {
    URL.revokeObjectURL(presetDownloadUrl);
    presetDownloadUrl = null;
  }
}

function resetColorDownloads() {
  colorDownloadUrls.forEach((url) => URL.revokeObjectURL(url));
  colorDownloadUrls = [];
  if (colorDownloadsEl) {
    colorDownloadsEl.innerHTML = "";
    colorDownloadsEl.style.display = "none";
  }
}

function renderColorDownloads(colors, baseName, presetPayload) {
  resetColorDownloads();
  if (!colorDownloadsEl || !Array.isArray(colors) || colors.length === 0) {
    return;
  }

  const cards = [];
  colors.forEach((info, idx) => {
    if (!info || !info.svg) {
      return;
    }
    const card = document.createElement("div");
    card.className = "color-card";

    const infoBox = document.createElement("div");
    infoBox.className = "color-info";
    const swatch = document.createElement("span");
    swatch.className = "color-swatch";
    swatch.style.backgroundColor = info.color || "#000000";
    infoBox.appendChild(swatch);

    const label = document.createElement("span");
    label.textContent = `${info.color || "#000000"} - ${info.final_points} pts`;
    infoBox.appendChild(label);
    card.appendChild(infoBox);

    const link = document.createElement("a");
    link.textContent = "Download";
    link.className = "color-download-link";
    const svgWithMetadata = injectSvgMetadata(info.svg, presetPayload, {
      output_type: "single_color",
      color: info.color || "",
    });
    const url = URL.createObjectURL(
      new Blob([svgWithMetadata], { type: "image/svg+xml" })
    );
    colorDownloadUrls.push(url);
    link.href = url;
    const suffix = (info.color || "").replace("#", "") || `color-${idx + 1}`;
    const base =
      baseName && baseName.trim().length > 0 ? baseName.trim() : "stitch";
    link.download = `${base}-${suffix}.svg`;

    card.appendChild(link);
    cards.push(card);
  });

  if (cards.length === 0) {
    return;
  }

  const fragment = document.createDocumentFragment();
  const heading = document.createElement("p");
  heading.className = "color-downloads-title";
  heading.textContent = "Download SVG per colore";
  fragment.appendChild(heading);
  cards.forEach((card) => fragment.appendChild(card));

  colorDownloadsEl.appendChild(fragment);
  colorDownloadsEl.style.display = "flex";
}

function renderPresetDownload(presetPayload, baseName) {
  if (!presetDownloadEl) {
    return;
  }
  if (presetDownloadUrl) {
    URL.revokeObjectURL(presetDownloadUrl);
  }
  const json = JSON.stringify(presetPayload, null, 2);
  presetDownloadUrl = URL.createObjectURL(
    new Blob([json], { type: "application/json" })
  );
  presetDownloadEl.href = presetDownloadUrl;
  presetDownloadEl.download = `${baseName}-params.json`;
  presetDownloadEl.style.display = "inline-block";
}

function imageDataUrl(base64) {
  return `data:image/png;base64,${base64}`;
}

function renderPreview(result) {
  if (!previewPanelEl || !result) {
    return;
  }

  previewPanelEl.style.display = "block";
  if (previewImageEl && result.overlay_png_base64) {
    previewImageEl.src = imageDataUrl(result.overlay_png_base64);
  }
  if (previewMaskEl && result.mask_png_base64) {
    previewMaskEl.src = imageDataUrl(result.mask_png_base64);
  }
  if (previewPointsEl && result.points_png_base64) {
    previewPointsEl.src = imageDataUrl(result.points_png_base64);
  }
  if (previewStatsEl) {
    previewStatsEl.textContent =
      `${result.width}x${result.height}px - ` +
      `${result.selected_pixels} pixel selezionati ` +
      `(${Number(result.selected_pct || 0).toFixed(1)}%) - ` +
      `${result.preview_points || 0} punti preview`;
  }
  if (!previewColorsEl) {
    return;
  }

  previewColorsEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  selectedPreviewColor = null;

  const allCard = document.createElement("div");
  allCard.className = "preview-color-card selected";
  allCard.dataset.color = "";
  const allMask = document.createElement("div");
  allMask.className = "preview-color-mask";
  allMask.style.cursor = "default";
  allCard.appendChild(allMask);
  const allMeta = document.createElement("div");
  allMeta.className = "preview-color-meta";
  const allTitle = document.createElement("strong");
  allTitle.textContent = "Tutti i colori";
  allMeta.appendChild(allTitle);
  const allDetails = document.createElement("span");
  allDetails.textContent = "SVG completo";
  allMeta.appendChild(allDetails);
  const allActions = document.createElement("div");
  allActions.className = "preview-color-actions";
  const allButton = document.createElement("button");
  allButton.type = "button";
  allButton.className = "btn-secondary";
  allButton.dataset.selectColor = "";
  allButton.textContent = "Usa tutti";
  allActions.appendChild(allButton);
  allMeta.appendChild(allActions);
  allCard.appendChild(allMeta);
  fragment.appendChild(allCard);

  (result.colors || []).forEach((info) => {
    const card = document.createElement("div");
    card.className = "preview-color-card";
    card.dataset.color = info.color || "#000000";

    const mask = document.createElement("img");
    mask.className = "preview-color-mask";
    mask.alt = `Maschera ${info.color}`;
    mask.src = imageDataUrl(info.mask_png_base64);
    card.appendChild(mask);

    const meta = document.createElement("div");
    meta.className = "preview-color-meta";

    const line = document.createElement("div");
    line.className = "preview-color-line";
    const swatch = document.createElement("span");
    swatch.className = "color-swatch";
    swatch.style.backgroundColor = info.color || "#000000";
    line.appendChild(swatch);
    const colorText = document.createElement("strong");
    colorText.textContent = info.color || "#000000";
    line.appendChild(colorText);
    meta.appendChild(line);

    const details = document.createElement("span");
    details.textContent =
      `${info.pixel_count} px - ${Number(info.area_pct || 0).toFixed(2)}%`;
    meta.appendChild(details);
    const actions = document.createElement("div");
    actions.className = "preview-color-actions";
    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "btn-secondary";
    selectButton.dataset.selectColor = info.color || "#000000";
    selectButton.textContent = "Solo SVG";
    actions.appendChild(selectButton);
    meta.appendChild(actions);
    card.appendChild(meta);
    fragment.appendChild(card);
  });
  previewColorsEl.appendChild(fragment);
}

function selectPreviewColor(color) {
  selectedPreviewColor = color || null;
  if (!previewColorsEl) {
    return;
  }
  previewColorsEl.querySelectorAll(".preview-color-card").forEach((card) => {
    const cardColor = card.dataset.color || "";
    card.classList.toggle("selected", cardColor === (selectedPreviewColor || ""));
  });
  if (statusEl) {
    statusEl.innerText = selectedPreviewColor
      ? `Colore selezionato per SVG: ${selectedPreviewColor}`
      : "Generazione SVG impostata su tutti i colori.";
  }
}

function openLightbox(src, alt = "Preview ingrandita") {
  if (!lightboxEl || !lightboxImageEl || !src) {
    return;
  }
  lightboxImageEl.src = src;
  lightboxImageEl.alt = alt;
  lightboxEl.classList.add("open");
  lightboxEl.setAttribute("aria-hidden", "false");
}

function closeLightbox() {
  if (!lightboxEl || !lightboxImageEl) {
    return;
  }
  lightboxEl.classList.remove("open");
  lightboxEl.setAttribute("aria-hidden", "true");
  lightboxImageEl.removeAttribute("src");
}

function createJobId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `job-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`;
}

function parseProgressParts(text) {
  if (!String(text).startsWith("__PROGRESS__|")) {
    return null;
  }
  const parts = String(text).split("|");
  return {
    pct: parts.length > 1 ? parseFloat(parts[1]) : 0,
    label: parts.length > 2 ? parts.slice(2).join("|") : "",
  };
}

function startLocalProgressPolling(jobId, pushStatus) {
  if (!jobId) {
    return () => {};
  }
  let lastMessageCount = 0;
  const startedAt = Date.now();
  let stopped = false;

  async function tick() {
    if (stopped) {
      return;
    }
    try {
      const result = await fetch(`/progress?job_id=${encodeURIComponent(jobId)}`)
        .then((res) => res.json());
      const messages = Array.isArray(result.messages) ? result.messages : [];
      messages.slice(lastMessageCount).forEach((msg) => {
        parseProgressMessage(String(msg), pushStatus);
      });
      lastMessageCount = messages.length;

      const pct = Number(result.progress_percent || 0);
      if (pct > 0 && pct < 100) {
        const elapsed = (Date.now() - startedAt) / 1000;
        const eta = elapsed * (100 / pct - 1);
        const label = result.progress_label || "Elaborazione";
        setProgress(pct, `${label} - ETA ${formatDuration(eta)}`);
      }
      if (result.done) {
        stopped = true;
        clearInterval(intervalId);
      }
    } catch (_) {
      // La fetch principale gestira eventuali errori del backend.
    }
  }

  const intervalId = setInterval(tick, 1000);
  tick();
  return () => {
    stopped = true;
    clearInterval(intervalId);
  };
}

async function handlePreview() {
  console.log("Analizza preview premuto");
  lastPreviewCacheKey = null;
  const statusHistory = [];
  const pushStatus = (msg) => {
    statusHistory.push(msg);
    statusEl.innerText = statusHistory.join("\n");
  };
  setProgress(0, "In attesa");

  const fileInput = document.getElementById("image");
  if (fileInput.files.length === 0) {
    pushStatus("Seleziona un'immagine prima di procedere.");
    return;
  }

  const file = fileInput.files[0];
  pushStatus("Analisi immagine...");
  setProgress(15, "Analisi preview");
  const arrayBuffer = await file.arrayBuffer();
  const imageBytes = new Uint8Array(arrayBuffer);
  const options = readOptions();

  try {
    const result = canUseLocalBackend()
      ? await runPreviewWithLocalBackend(imageBytes, options, pushStatus)
      : await runPreviewWithPyodide(imageBytes, options, pushStatus);
    lastPreviewCacheKey = result.preview_cache_key || null;
    renderPreview(result);
    setProgress(100, "Preview pronta");
    pushStatus(
      `Preview pronta: ${result.selected_pixels} pixel, ` +
        `${(result.colors || []).length} colori.`
    );
  } catch (error) {
    console.error("Errore durante la preview:", error);
    setProgress(100, "Errore");
    pushStatus(`Errore preview: ${error.message || error}`);
  }
}

async function handleConvert() {
  console.log("Genera SVG premuto");
  if (liveProgressStop) {
    liveProgressStop();
    liveProgressStop = null;
  }
  resetPrimaryDownloads();
  resetColorDownloads();
  const statusHistory = [];
  const seenStatus = new Set();
  const pushStatus = (msg) => {
    const text = String(msg || "");
    if (!text || seenStatus.has(text)) {
      return;
    }
    seenStatus.add(text);
    statusHistory.push(text);
    if (statusHistory.length > 140) {
      statusHistory.splice(0, statusHistory.length - 140);
    }
    statusEl.innerText = statusHistory.join("\n");
  };
  setProgress(0, "In attesa");

  const fileInput = document.getElementById("image");
  if (fileInput.files.length === 0) {
    pushStatus("Seleziona un'immagine prima di procedere.");
    return;
  }

  const file = fileInput.files[0];
  pushStatus("Lettura immagine...");
  setProgress(20, "Lettura immagine");
  const arrayBuffer = await file.arrayBuffer();
  const imageBytes = new Uint8Array(arrayBuffer);
  const options = readOptions();
  const runOptions = { ...options };
  if (lastPreviewCacheKey) {
    runOptions._preview_cache_key = lastPreviewCacheKey;
  }
  if (selectedPreviewColor) {
    runOptions._only_color = selectedPreviewColor;
  }
  const useLocalBackend = canUseLocalBackend();
  const jobId = useLocalBackend ? createJobId() : "";
  if (jobId) {
    runOptions._job_id = jobId;
  }

  try {
    pushStatus("Elaborazione in corso...");
    setProgress(25, "Elaborazione");
    if (jobId) {
      liveProgressStop = startLocalProgressPolling(jobId, pushStatus);
    }
    const result = useLocalBackend
      ? await runWithLocalBackend(imageBytes, runOptions, pushStatus, !jobId)
      : await runWithPyodide(imageBytes, runOptions, pushStatus);
    if (liveProgressStop) {
      liveProgressStop();
      liveProgressStop = null;
    }

    console.log("Result summary:", result.summary);

    const svgStr = result.svg;
    const summary = result.summary;
    console.log("SVG preview:", svgStr.slice(0, 120));

    const baseName = sanitizeFileStem(file.name);
    const presetPayload = buildPresetPayload(options, file, result);

    const multiSvg = injectSvgMetadata(svgStr, presetPayload, {
      output_type: "multi_color",
    });
    const multiBlob = new Blob([multiSvg], { type: "image/svg+xml" });
    const multiUrl = URL.createObjectURL(multiBlob);
    multiDownloadUrl = multiUrl;
    if (multiDownloadEl) {
      multiDownloadEl.href = multiUrl;
      multiDownloadEl.download = `${baseName}-multi.svg`;
      multiDownloadEl.style.display = "inline-block";
    }

    const monoSvg =
      typeof result.mono_svg === "string" && result.mono_svg.trim().length > 0
        ? result.mono_svg
        : svgStr;
    const monoSvgWithMetadata = injectSvgMetadata(monoSvg, presetPayload, {
      output_type: "mono",
    });
    const monoBlob = new Blob([monoSvgWithMetadata], { type: "image/svg+xml" });
    monoDownloadUrl = URL.createObjectURL(monoBlob);
    downloadEl.href = monoDownloadUrl;
    downloadEl.download = `${baseName}-mono.svg`;
    downloadEl.style.display = "inline-block";

    renderColorDownloads(result.colors || [], baseName, presetPayload);
    renderPresetDownload(presetPayload, baseName);
    setProgress(100, "Completato");
    pushStatus(summary);
  } catch (error) {
    if (liveProgressStop) {
      liveProgressStop();
      liveProgressStop = null;
    }
    console.error("Errore durante la conversione:", error);
    setProgress(100, "Errore");
    pushStatus(`Errore: ${error.message || error}`);
  }
}

async function handlePresetLoad(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }
  try {
    const text = await file.text();
    const preset = parsePresetText(text);
    const options = preset.options || preset;
    applyOptions(options);
    statusEl.innerText =
      `Preset caricato: ${file.name}\n` +
      "Puoi analizzare la preview o generare un nuovo SVG con questi parametri.";
  } catch (error) {
    statusEl.innerText = `Errore preset: ${error.message || error}`;
  } finally {
    event.target.value = "";
  }
}

function parseProgressMessage(text, pushStatus) {
  const progress = parseProgressParts(text);
  if (progress) {
    setProgress(progress.pct, progress.label);
    return;
  }
  pushStatus(text);
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

async function runWithLocalBackend(imageBytes, options, pushStatus, replayLogs = true) {
  pushStatus("Uso motore Python locale...");
  let response;
  try {
    response = await fetch("/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: bytesToBase64(imageBytes),
        options,
      }),
    });
  } catch (error) {
    let health = null;
    try {
      health = await fetch("/health").then((res) => res.json());
    } catch (_) {
      // Il server non risponde: probabilmente il processo locale si e' chiuso.
    }
    const logPath = health && health.log_file ? `\nLog: ${health.log_file}` : "";
    throw new Error(
      "Il motore locale ha chiuso la connessione durante la conversione. " +
        "Riavvia BitmapToStitch.exe e riprova con max-width o max-points piu bassi." +
        logPath
    );
  }
  const result = await response.json();
  if (!response.ok) {
    const logPath = result.log_file ? `\nLog: ${result.log_file}` : "";
    throw new Error(
      (result.error || "Conversione locale non riuscita.") + logPath
    );
  }
  if (replayLogs) {
    (result.logs || []).forEach((msg) =>
      parseProgressMessage(String(msg), pushStatus)
    );
  }
  return result;
}

async function runPreviewWithLocalBackend(imageBytes, options, pushStatus) {
  pushStatus("Uso motore Python locale per la preview...");
  let response;
  try {
    response = await fetch("/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: bytesToBase64(imageBytes),
        options,
      }),
    });
  } catch (error) {
    throw new Error(
      "Preview locale non riuscita. Riavvia BitmapToStitch e riprova."
    );
  }
  const result = await response.json();
  if (!response.ok) {
    const logPath = result.log_file ? `\nLog: ${result.log_file}` : "";
    throw new Error((result.error || "Preview locale non riuscita.") + logPath);
  }
  (result.logs || []).forEach((msg) =>
    parseProgressMessage(String(msg), pushStatus)
  );
  return result;
}

async function runWithPyodide(imageBytes, options, pushStatus) {
  pushStatus("Carico runtime Python...");
  setProgress(4, "Inizializzo");
  const pyodide = await initPyodide();
  const reportStatus = pyodide.toPy((msg) => {
    parseProgressMessage(String(msg || ""), pushStatus);
  });
  pyodide.globals.set("status_callback", reportStatus);

  try {
    const runPipeline = pyodide.globals.get("run_pipeline_browser");
    const pyBytes = pyodide.toPy(imageBytes);
    const pyOptions = pyodide.toPy(options);
    const resultProxy = runPipeline(pyBytes, pyOptions);
    const result = resultProxy.toJs({
      create_proxies: false,
      dict_converter: Object.fromEntries,
    });
    resultProxy.destroy();
    pyBytes.destroy();
    pyOptions.destroy();
    runPipeline.destroy();
    return result;
  } finally {
    pyodide.runPython("status_callback = None");
    if (typeof reportStatus.destroy === "function") {
      reportStatus.destroy();
    }
  }
}

async function runPreviewWithPyodide(imageBytes, options, pushStatus) {
  pushStatus("Carico runtime Python...");
  setProgress(4, "Inizializzo");
  const pyodide = await initPyodide();
  const runPreview = pyodide.globals.get("analyze_preview_browser");
  const pyBytes = pyodide.toPy(imageBytes);
  const pyOptions = pyodide.toPy(options);
  try {
    const resultProxy = runPreview(pyBytes, pyOptions);
    const result = resultProxy.toJs({
      create_proxies: false,
      dict_converter: Object.fromEntries,
    });
    resultProxy.destroy();
    return result;
  } finally {
    pyBytes.destroy();
    pyOptions.destroy();
    runPreview.destroy();
  }
}

previewBtn.addEventListener("click", () => {
  handlePreview();
});

if (previewPanelEl) {
  previewPanelEl.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof HTMLButtonElement && "selectColor" in target.dataset) {
      selectPreviewColor(target.dataset.selectColor || "");
      return;
    }
    if (target instanceof HTMLImageElement && target.src) {
      openLightbox(target.src, target.alt || "Preview ingrandita");
    }
  });
}

if (closeLightboxBtn) {
  closeLightboxBtn.addEventListener("click", closeLightbox);
}

if (lightboxEl) {
  lightboxEl.addEventListener("click", (event) => {
    if (event.target === lightboxEl) {
      closeLightbox();
    }
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && lightboxEl?.classList.contains("open")) {
    closeLightbox();
  }
});

if (presetInputEl) {
  presetInputEl.addEventListener("change", handlePresetLoad);
}

convertBtn.addEventListener("click", () => {
  handleConvert();
});
