from __future__ import annotations

import json
import math
import mimetypes
import shutil
import subprocess
import threading
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
UI_ROOT = PROJECT_ROOT / "ui"
MAX_CAPTIONS = 7

SPEAKERS = (
    {"id": 1, "name": "Speaker 1", "color": "#2878f0"},
    {"id": 2, "name": "Speaker 2", "color": "#20b45a"},
    {"id": 3, "name": "Speaker 3", "color": "#ff9818"},
)


@dataclass(frozen=True)
class Caption:

    speaker_id: int
    text: str
    time: str
    latency_ms: float


class RuntimeState:

    def __init__(self, demo: bool = False) -> None:
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._frame: bytes | None = None
        self._frame_version = 0
        self._faces: list[dict[str, Any]] = []
        self._active_speaker_id: int | None = None
        self._direction = (1.0, 0.0, 0.0)
        self._captions: deque[Caption] = deque(maxlen=MAX_CAPTIONS)
        self._partial: dict[str, Any] | None = None
        self._utterance_speakers: dict[int, int] = {}
        self._status: dict[str, Any] = {
            "demo": demo,
            "recording": True,
            "diarization": True,
            "model": True,
            "stt_raw": False,
            "camera_connected": False,
            "audio_connected": False,
            "doa_ready": False,
            "stt_ready": demo,
            "error": None,
        }

    def set_status(self, **changes: Any) -> None:

        with self._lock:
            self._status.update(changes)

    def toggle(self, key: str) -> bool:

        if key not in {"recording", "diarization", "model", "stt_raw"}:
            raise ValueError(f"unsupported control: {key}")
        with self._lock:
            self._status[key] = not bool(self._status[key])
            if key == "recording" and not self._status[key]:
                self._partial = None
            return bool(self._status[key])

    def clear_captions(self) -> None:

        with self._lock:
            self._captions.clear()
            self._partial = None
            self._utterance_speakers.clear()

    def is_recording(self) -> bool:

        with self._lock:
            return bool(self._status["recording"])

    def set_direction(self, direction: tuple[float, float, float]) -> None:

        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-9:
            return
        with self._lock:
            self._direction = tuple(value / norm for value in direction)
            self._status["doa_ready"] = True

    def direction(self) -> tuple[float, float, float]:

        with self._lock:
            return tuple(self._direction)

    def set_frame(self, jpeg: bytes) -> None:

        with self._frame_condition:
            self._frame = jpeg
            self._frame_version += 1
            self._frame_condition.notify_all()

    def wait_for_frame(
        self,
        previous_version: int,
        timeout: float,
    ) -> tuple[bytes | None, int]:

        with self._frame_condition:
            if self._frame_version == previous_version:
                self._frame_condition.wait(timeout=timeout)
            return self._frame, self._frame_version

    def update_faces(
        self,
        faces: list[dict[str, Any]],
        active_speaker_id: int | None,
    ) -> None:

        with self._lock:
            self._faces = faces
            self._active_speaker_id = active_speaker_id

    def active_speaker_id(self) -> int:

        with self._lock:
            return self._active_speaker_id or 0

    def handle_stt_result(self, result: Any) -> None:

        with self._lock:
            result_type = getattr(result, "type", "error")
            utterance_id = int(getattr(result, "utterance_id", 0))
            speaker_id = self._utterance_speakers.setdefault(
                utterance_id,
                self._active_speaker_id or 0,
            )

            if result_type == "partial":
                self._partial = {
                    "speaker_id": speaker_id,
                    "text": str(getattr(result, "text", "")),
                    "latency_ms": float(getattr(result, "latency_ms", 0.0)),
                }
                return

            if result_type == "final":
                text = str(getattr(result, "text", "")).strip()
                if text:
                    self._captions.append(
                        Caption(
                            speaker_id=speaker_id,
                            text=text,
                            time=datetime.now().astimezone().strftime("%H:%M:%S"),
                            latency_ms=round(
                                float(getattr(result, "latency_ms", 0.0)),
                                2,
                            ),
                        )
                    )
                self._partial = None
                self._utterance_speakers.pop(utterance_id, None)
                return

            error = str(getattr(result, "error", "unknown STT error"))
            self._status["error"] = error
            if bool(getattr(result, "is_final", False)):
                self._partial = None
                self._utterance_speakers.pop(utterance_id, None)

    def snapshot(self) -> dict[str, Any]:

        with self._lock:
            return {
                "speakers": list(SPEAKERS),
                "faces": [dict(face) for face in self._faces],
                "active_speaker_id": self._active_speaker_id,
                "captions": [asdict(caption) for caption in self._captions],
                "partial": dict(self._partial) if self._partial else None,
                "status": dict(self._status),
            }


class CaptionHTTPServer(ThreadingHTTPServer):

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        state: RuntimeState,
        stop_event: threading.Event,
        control_hook: Any = None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.state = state
        self.stop_event = stop_event

        self.control_hook = control_hook


class UIRequestHandler(BaseHTTPRequestHandler):

    server: CaptionHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.server.state.snapshot())
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/camera.mjpg":
            self._stream_camera()
            return

        relative = "index.html" if path == "/" else path.lstrip("/")
        if relative not in {"index.html", "styles.css", "app.js"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(UI_ROOT / relative)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = str(payload.get("action", ""))
            if action == "reset_captions":
                self.server.state.clear_captions()
                value = True
            else:
                value = self.server.state.toggle(action)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        hook = getattr(self.server, "control_hook", None)
        if hook is not None:
            try:
                hook(action, value)
            except Exception:
                pass
        self._send_json({"ok": True, "value": value})

    def log_message(self, format_string: str, *args: Any) -> None:
        del format_string, args

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _stream_camera(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        try:
            while not self.server.stop_event.is_set():
                frame, version = self.server.state.wait_for_frame(version, timeout=1.0)
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def open_kiosk(url: str) -> subprocess.Popen[Any] | None:

    browser = next(
        (
            executable
            for name in (
                "chromium-browser",
                "chromium",
                "google-chrome",
                "google-chrome-stable",
            )
            if (executable := shutil.which(name))
        ),
        None,
    )
    if browser is None:
        print(f"[browser] Chromium not found, open {url} manually")
        webbrowser.open(url)
        return None
    return subprocess.Popen(
        [
            browser,
            "--kiosk",
            "--start-fullscreen",
            "--window-size=1024,600",
            "--window-position=0,0",
            "--force-device-scale-factor=1",
            "--disable-pinch",
            "--noerrdialogs",
            "--disable-infobars",
            "--disable-session-crashed-bubble",
            "--disable-features=TranslateUI",
            f"--app={url}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
