const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download-link");
const convertBtn = document.getElementById("convert");
let pyodideReady = null;

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
  };
}

async function handleConvert() {
  console.log("Genera SVG premuto");
  downloadEl.style.display = "none";
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

    const blob = new Blob([svgStr], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    downloadEl.href = url;
    downloadEl.download = `${file.name.replace(/\.[^.]+$/, "") || "stitch"}.svg`;
    downloadEl.style.display = "inline-block";

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
