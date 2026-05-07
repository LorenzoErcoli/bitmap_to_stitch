const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download-link");
const multiDownloadEl = document.getElementById("download-multi");
const colorDownloadsEl = document.getElementById("color-downloads");
const progressBarEl = document.getElementById("progress-bar");
const progressTextEl = document.getElementById("progress-text");
const convertBtn = document.getElementById("convert");
let pyodideReady = null;
let colorDownloadUrls = [];
let monoDownloadUrl = null;
let multiDownloadUrl = null;

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
}

function resetColorDownloads() {
  colorDownloadUrls.forEach((url) => URL.revokeObjectURL(url));
  colorDownloadUrls = [];
  if (colorDownloadsEl) {
    colorDownloadsEl.innerHTML = "";
    colorDownloadsEl.style.display = "none";
  }
}

function renderColorDownloads(colors, baseName) {
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
    const url = URL.createObjectURL(
      new Blob([info.svg], { type: "image/svg+xml" })
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

async function handleConvert() {
  console.log("Genera SVG premuto");
  resetPrimaryDownloads();
  resetColorDownloads();
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
  pushStatus("Lettura immagine...");
  setProgress(20, "Lettura immagine");
  const arrayBuffer = await file.arrayBuffer();
  const imageBytes = new Uint8Array(arrayBuffer);
  const options = readOptions();

  try {
    pushStatus("Elaborazione in corso...");
    setProgress(25, "Elaborazione");
    const result = canUseLocalBackend()
      ? await runWithLocalBackend(imageBytes, options, pushStatus)
      : await runWithPyodide(imageBytes, options, pushStatus);

    console.log("Result summary:", result.summary);

    const svgStr = result.svg;
    const summary = result.summary;
    console.log("SVG preview:", svgStr.slice(0, 120));

    const baseName = file.name.replace(/\.[^.]+$/, "") || "stitch";

    const multiBlob = new Blob([svgStr], { type: "image/svg+xml" });
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
    const monoBlob = new Blob([monoSvg], { type: "image/svg+xml" });
    monoDownloadUrl = URL.createObjectURL(monoBlob);
    downloadEl.href = monoDownloadUrl;
    downloadEl.download = `${baseName}-mono.svg`;
    downloadEl.style.display = "inline-block";

    renderColorDownloads(result.colors || [], baseName);
    setProgress(100, "Completato");
    pushStatus(summary);
  } catch (error) {
    console.error("Errore durante la conversione:", error);
    setProgress(100, "Errore");
    pushStatus(`Errore: ${error.message || error}`);
  }
}

function parseProgressMessage(text, pushStatus) {
  if (text.startsWith("__PROGRESS__|")) {
    const parts = text.split("|");
    const pct = parts.length > 1 ? parseFloat(parts[1]) : 0;
    const label = parts.length > 2 ? parts.slice(2).join("|") : "";
    setProgress(pct, label);
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

async function runWithLocalBackend(imageBytes, options, pushStatus) {
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
  (result.logs || []).forEach((msg) => parseProgressMessage(String(msg), pushStatus));
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
    reportStatus.destroy();
  }
}

convertBtn.addEventListener("click", () => {
  handleConvert();
});
