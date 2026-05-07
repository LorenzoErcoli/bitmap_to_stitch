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
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


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


def log_event(message):
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line)
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


class LocalAppHandler(BaseHTTPRequestHandler):
    server_version = "BitmapToStitchLocal/0.1.1"

    def log_message(self, fmt, *args):
        log_event("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path == "/health":
            _json_response(
                self,
                200,
                {"ok": True, "mode": "local", "log_file": str(LOG_FILE)},
            )
            return

        rel_path = unquote(self.path.split("?", 1)[0])
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
        if self.path != "/convert":
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

            log_event(
                "Conversione richiesta: "
                f"payload_base64_chars={len(image_b64)}, "
                f"ordering={options.get('ordering')}, "
                f"max_width={options.get('max_width')}, "
                f"max_points={options.get('max_points')}"
            )
            image_bytes = base64.b64decode(image_b64, validate=True)
            logs = []
            pipeline_app.status_callback = lambda msg: logs.append(str(msg))
            try:
                result = pipeline_app.run_pipeline_browser(image_bytes, options)
            finally:
                pipeline_app.status_callback = None

            result["logs"] = logs
            log_event(
                "Conversione completata: "
                f"svg_chars={len(result.get('svg', ''))}, "
                f"colors={len(result.get('colors', []))}"
            )
            _json_response(self, 200, result)
        except BaseException as exc:
            pipeline_app.status_callback = None
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
    selected_port = DEFAULT_PORT
    should_open_browser = True
    if len(sys.argv) > 1:
        selected_port = int(sys.argv[1])
    if "--no-browser" in sys.argv:
        should_open_browser = False
    run(selected_port, should_open_browser)
