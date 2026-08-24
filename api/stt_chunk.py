"""Chunked streaming STT for live-call voice mode.

The client streams short audio chunks (2-3s) to /api/voice/stt-chunk. The
server accumulates them per session_id and returns partial transcripts, so
the app can show "listening..." text as the user speaks (like a live call),
instead of waiting for a full utterance file.

POST /api/voice/stt-chunk
  multipart: file=<audio chunk>, session=<id>
  → { ok, partial: "text so far", final: false }

POST /api/voice/stt-final
  multipart: file=<audio>, session=<id>
  → { ok, transcript: "full text", final: true }
  (clears the session buffer)
"""

import tempfile
import threading
import time
from pathlib import Path

# Per-session rolling buffers: session_id -> {"chunks": [bytes], "text": str, "ts": float}
_BUFFERS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MAX_CHUNKS = 40  # ~2min of 3s chunks
_BUFFER_TTL = 120.0  # seconds; stale buffers get dropped

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _get_buffer(session: str) -> dict:
    now = time.time()
    with _LOCK:
        buf = _BUFFERS.get(session)
        if buf is None or now - buf["ts"] > _BUFFER_TTL:
            buf = {"chunks": [], "text": "", "ts": now}
            _BUFFERS[session] = buf
        buf["ts"] = now
        return buf


def _clear_buffer(session: str):
    with _LOCK:
        _BUFFERS.pop(session, None)


def _transcribe_bytes(audio_bytes: bytes, suffix: str) -> str:
    """Run faster-whisper (or configured STT) on raw audio bytes."""
    from tools.transcription_tools import transcribe_audio

    with tempfile.NamedTemporaryFile(prefix="webui-stt-chunk-", suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = transcribe_audio(tmp_path)
        if result.get("success"):
            return str(result.get("transcript") or "").strip()
        return ""
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _parse_multipart(handler):
    """Parse multipart form from the request. Returns (fields, files)."""
    content_type = handler.headers.get("Content-Type", "")
    content_length = int(handler.headers.get("Content-Length", 0) or 0)
    if content_length > _MAX_UPLOAD_BYTES:
        raise ValueError("file too large")
    from api.upload import parse_multipart

    fields, files = parse_multipart(handler.rfile, content_type, content_length)
    return fields, files


def handle_stt_chunk(handler, parsed):
    """Accumulate an audio chunk and return a partial transcript."""
    import json as _json
    from urllib.parse import parse_qs as _parse_qs

    qs = _parse_qs(parsed.query)
    session = (qs.get("session", [""])[0] or "").strip() or "default"
    try:
        fields, files = _parse_multipart(handler)
    except Exception as e:
        return _json_response(handler, {"error": str(e)}, 400)

    if "file" not in files:
        return _json_response(handler, {"error": "No file field"}, 400)
    filename, file_bytes = files["file"]
    suffix = Path(filename or "clip.webm").suffix or ".webm"

    buf = _get_buffer(session)
    buf["chunks"].append(file_bytes)
    if len(buf["chunks"]) > _MAX_CHUNKS:
        buf["chunks"] = buf["chunks"][-_MAX_CHUNKS:]

    # Transcribe the accumulated audio (last few chunks for speed)
    combined = b"".join(buf["chunks"][-4:])
    text = _transcribe_bytes(combined, suffix)
    if text:
        buf["text"] = text

    return _json_response(handler, {"ok": True, "partial": buf["text"], "final": False})


def handle_stt_final(handler, parsed):
    """Finalize: transcribe the full buffer and clear it."""
    import json as _json
    from urllib.parse import parse_qs as _parse_qs

    qs = _parse_qs(parsed.query)
    session = (qs.get("session", [""])[0] or "").strip() or "default"
    try:
        fields, files = _parse_multipart(handler)
    except Exception as e:
        return _json_response(handler, {"error": str(e)}, 400)

    if "file" not in files:
        return _json_response(handler, {"error": "No file field"}, 400)
    filename, file_bytes = files["file"]
    suffix = Path(filename or "clip.webm").suffix or ".webm"

    buf = _get_buffer(session)
    buf["chunks"].append(file_bytes)
    combined = b"".join(buf["chunks"])
    text = _transcribe_bytes(combined, suffix)
    _clear_buffer(session)

    return _json_response(handler, {"ok": True, "transcript": text, "final": True})


def _json_response(handler, obj, status=200):
    import json as _json

    body = _json.dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True
