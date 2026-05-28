#!/usr/bin/env python3
import base64
import datetime as _dt
import faulthandler
import json
import mimetypes
import os
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


if getattr(sys, "frozen", False):
    ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
APP_NAME = "BitmapToStitch"


def _app_data_dir():
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / f".{APP_NAME}"


def _select_log_dir():
    candidates = [_app_data_dir()]
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(tempfile.gettempdir()) / APP_NAME)
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path.cwd()


LOG_DIR = _select_log_dir()
LOG_FILE = LOG_DIR / "app.log"
PROGRESS_JOBS = {}
PROGRESS_LOCK = threading.Lock()
MAX_PROGRESS_MESSAGES = 300


def log_event(message):
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    try:
        print(line)
    except OSError:
        pass
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


_fault_log = LOG_FILE.open("a", encoding="utf-8")
faulthandler.enable(file=_fault_log, all_threads=True)


sys.path.insert(0, str(WEB_ROOT))
import app as pipeline_app  # noqa: E402


def _json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _update_progress_job(job_id, **updates):
    if not job_id:
        return
    with PROGRESS_LOCK:
        job = PROGRESS_JOBS.setdefault(
            job_id,
            {
                "job_id": job_id,
                "started_at": time.time(),
                "updated_at": time.time(),
                "progress_percent": 0.0,
                "progress_label": "",
                "messages": [],
                "done": False,
                "error": "",
            },
        )
        if "message" in updates:
            job["messages"].append(str(updates.pop("message")))
            if len(job["messages"]) > MAX_PROGRESS_MESSAGES:
                job["messages"] = job["messages"][-MAX_PROGRESS_MESSAGES:]
        job.update(updates)
        job["updated_at"] = time.time()


def _read_progress_job(job_id):
    with PROGRESS_LOCK:
        job = PROGRESS_JOBS.get(job_id)
        if not job:
            return None
        payload = dict(job)
        payload["messages"] = list(job.get("messages", []))
        payload["elapsed_s"] = max(0.0, time.time() - float(job.get("started_at", time.time())))
        return payload


class LocalAppHandler(BaseHTTPRequestHandler):
    server_version = "BitmapToStitchLocal/0.1.4"

    def log_message(self, fmt, *args):
        log_event("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            _json_response(
                self,
                200,
                {"ok": True, "mode": "local", "log_file": str(LOG_FILE)},
            )
            return

        if parsed.path == "/progress":
            query = parse_qs(parsed.query)
            job_id = (query.get("job_id") or [""])[0]
            payload = _read_progress_job(job_id)
            if payload is None:
                payload = {"job_id": job_id, "messages": [], "done": False}
            _json_response(self, 200, payload)
            return

        rel_path = unquote(parsed.path)
        if rel_path in ("", "/"):
            rel_path = "/index.html"

        target = (WEB_ROOT / rel_path.lstrip("/")).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(403)
            return

        if not target.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in ("/convert", "/preview"):
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            image_b64 = payload.get("image_base64", "")
            options = payload.get("options", {})
            if not image_b64:
                raise ValueError("Immagine mancante.")

            is_preview = self.path == "/preview"
            log_event(
                ("Preview richiesta: " if is_preview else "Conversione richiesta: ")
                +
                f"payload_base64_chars={len(image_b64)}, "
                f"ordering={options.get('ordering')}, "
                f"max_width={options.get('max_width')}, "
                f"max_points={options.get('max_points')}"
            )
            image_bytes = base64.b64decode(image_b64, validate=True)
            logs = []
            job_id = str(options.get("_job_id") or "") if not is_preview else ""
            if job_id:
                _update_progress_job(
                    job_id,
                    progress_percent=0.0,
                    progress_label="Avvio conversione",
                    done=False,
                    error="",
                    messages=[],
                )

            def capture_pipeline_status(msg):
                text = str(msg)
                logs.append(text)
                progress_percent = None
                progress_label = ""
                if text.startswith("__PROGRESS__|"):
                    parts = text.split("|")
                    pct = parts[1] if len(parts) > 1 else "?"
                    label = parts[2] if len(parts) > 2 else ""
                    try:
                        progress_percent = float(pct)
                    except (TypeError, ValueError):
                        progress_percent = None
                    progress_label = label
                    log_event(f"Pipeline progress: {pct}% {label}".rstrip())
                else:
                    log_event(f"Pipeline: {text}")
                update = {"message": text}
                if progress_percent is not None:
                    update["progress_percent"] = progress_percent
                    update["progress_label"] = progress_label
                _update_progress_job(job_id, **update)

            pipeline_app.status_callback = capture_pipeline_status
            try:
                if is_preview:
                    result = pipeline_app.analyze_preview_browser(image_bytes, options)
                else:
                    result = pipeline_app.run_pipeline_browser(image_bytes, options)
            finally:
                pipeline_app.status_callback = None

            result["logs"] = logs
            if is_preview:
                log_event(
                    "Preview completata: "
                    f"selected_pixels={result.get('selected_pixels')}, "
                    f"colors={len(result.get('colors', []))}"
                )
            else:
                _update_progress_job(
                    job_id,
                    progress_percent=100.0,
                    progress_label="Conversione completata",
                    done=True,
                    message="Conversione completata.",
                )
                log_event(
                    "Conversione completata: "
                    f"svg_chars={len(result.get('svg', ''))}, "
                    f"colors={len(result.get('colors', []))}"
                )
            _json_response(self, 200, result)
        except BaseException as exc:
            pipeline_app.status_callback = None
            try:
                options = locals().get("options", {})
                job_id = str(options.get("_job_id") or "")
                _update_progress_job(
                    job_id,
                    done=True,
                    error=str(exc),
                    message=f"Errore: {exc}",
                )
            except Exception:
                pass
            log_event("Errore conversione:\n" + traceback.format_exc())
            _json_response(
                self,
                400,
                {
                    "error": str(exc),
                    "log_file": str(LOG_FILE),
                },
            )


def run(port=DEFAULT_PORT, open_browser=True):
    server = ThreadingHTTPServer((HOST, int(port)), LocalAppHandler)
    url = f"http://{HOST}:{int(port)}/"
    log_event(f"Bitmap to Stitch locale avviato: {url}")
    log_event(f"Log file: {LOG_FILE}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    try:
        selected_port = DEFAULT_PORT
        should_open_browser = True
        if len(sys.argv) > 1:
            selected_port = int(sys.argv[1])
        if "--no-browser" in sys.argv:
            should_open_browser = False
        run(selected_port, should_open_browser)
    except BaseException:
        log_event("Errore avvio server:\n" + traceback.format_exc())
        raise
