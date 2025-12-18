const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download-link");
const multiDownloadEl = document.getElementById("download-multi");
const colorDownloadsEl = document.getElementById("color-downloads");
const convertBtn = document.getElementById("convert");
let pyodideReady = null;
let colorDownloadUrls = [];
let monoDownloadUrl = null;
let multiDownloadUrl = null;

async function initPyodide() {
  if (!pyodideReady) {
    statusEl.textContent = "Carico runtime Python...";
    pyodideReady = (async () => {
      const pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
      });
      await pyodide.loadPackage(["numpy", "pillow"]);
      const appSource = await fetch("app.py").then((res) => res.text());
      await pyodide.runPythonAsync(appSource);
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
    max_points: num("max-points", 0, parseInt),
    scale: num("scale", 1.0),
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

  const fileInput = document.getElementById("image");
  if (fileInput.files.length === 0) {
    pushStatus("Seleziona un'immagine prima di procedere.");
    return;
  }

  const file = fileInput.files[0];
  pushStatus("Carico runtime Python...");
  const pyodide = await initPyodide();

  pushStatus("Lettura immagine...");
  const arrayBuffer = await file.arrayBuffer();
  const imageBytes = new Uint8Array(arrayBuffer);
  const options = readOptions();

  const reportStatus = pyodide.toPy((msg) => {
    pushStatus(msg);
  });
  pyodide.globals.set("status_callback", reportStatus);

  try {
    pushStatus("Elaborazione in corso...");
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
    pushStatus(summary);
  } catch (error) {
    console.error("Errore durante la conversione:", error);
    pushStatus(`Errore: ${error.message || error}`);
  } finally {
    pyodide.runPython("status_callback = None");
    reportStatus.destroy();
  }
}

convertBtn.addEventListener("click", () => {
  handleConvert();
});
