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
import os
import urllib.request
import urllib.error
from pathlib import Path

# Per-session rolling buffers: session_id -> {"chunks": [bytes], "text": str, "ts": float}
_BUFFERS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MAX_CHUNKS = 40  # ~2min of 3s chunks
_BUFFER_TTL = 120.0  # seconds; stale buffers get dropped

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _reap_buffers() -> None:
    """Drop stale session buffers that were never finalized (client crash)."""
    while True:
        time.sleep(60)
        try:
            now = time.time()
            with _LOCK:
                stale = [s for s, b in _BUFFERS.items() if now - b.get("ts", 0) > _BUFFER_TTL]
                for s in stale:
                    _BUFFERS.pop(s, None)
        except Exception:
            pass


_reaper = threading.Thread(target=_reap_buffers, daemon=True)
_reaper.start()



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


"""STT provider registry + kind dispatchers. Appended/spliced into stt_chunk.py."""

import json as _json_mod

_STT_CONFIG_PATH = Path(os.environ.get("NYX_STT_CONFIG", "/home/ubuntu/hermes-webui/api/stt_providers.json"))


def _load_custom_ids() -> set:
    """Return the set of custom (user-added) provider ids."""
    try:
        with open(_STT_CONFIG_PATH) as f:
            data = _json_mod.load(f)
        return set((data.get("custom") or {}).keys())
    except Exception:
        return set()


def _load_stt_providers() -> dict:
    """Load the STT provider registry from JSON. Returns {id: {...}}."""
    try:
        with open(_STT_CONFIG_PATH) as f:
            data = _json_mod.load(f)
        provs = data.get("providers", {})
        # merge custom providers persisted alongside the seed config
        custom = data.get("custom", {})
        provs.update(custom)
        return provs
    except Exception:
        return {}


def _enabled_providers() -> list:
    """Return providers with a usable key, sorted by priority ascending."""
    out = []
    for pid, p in _load_stt_providers().items():
        if not p.get("enabled", True):
            continue
        key = os.environ.get(p.get("api_key_env", ""), "")
        if not key:
            continue
        out.append((pid, p))
    out.sort(key=lambda x: x[1].get("priority", 99))
    return out


def _mime_for(suffix: str) -> str:
    return {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
        ".webm": "audio/webm", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".mp4": "audio/mp4",
    }.get(suffix.lower(), "application/octet-stream")


def _transcribe_openai_compat(audio_path: str, suffix: str, p: dict) -> str:
    """OpenAI-compatible /audio/transcriptions (Groq, OpenAI, local, BYO)."""
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        return ""
    base = (p.get("base_url") or "").rstrip("/")
    model = p.get("model") or os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    url = base + "/audio/transcriptions"
    MAX = 25 * 1024 * 1024
    with open(audio_path, "rb") as f:
        data = f.read()
    if len(data) > MAX:
        return ""
    import io
    import uuid
    boundary = "----stt-%s" % uuid.uuid4().hex
    mime = _mime_for(suffix)
    fname = "audio" + suffix
    crlf = b"\r\n"
    body = io.BytesIO()
    body.write(("--%s" % boundary).encode() + crlf)
    body.write(b'Content-Disposition: form-data; name="model"' + crlf + crlf)
    body.write(model.encode() + crlf)
    body.write(("--%s" % boundary).encode() + crlf)
    body.write(('Content-Disposition: form-data; name="file"; filename="%s"' % fname).encode() + crlf)
    body.write(("Content-Type: %s" % mime).encode() + crlf + crlf)
    body.write(data)
    body.write(crlf)
    body.write(("--%s--" % boundary).encode() + crlf)
    body_bytes = body.getvalue()
    req = urllib.request.Request(url, data=body_bytes, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "multipart/form-data; boundary=%s" % boundary,
        "Content-Length": str(len(body_bytes)),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print("[webui] stt openai_compat non-200: %s" % resp.status, flush=True)
                return ""
            out = _json_mod.loads(resp.read().decode("utf-8", "replace"))
            return str(out.get("text") or "").strip()
    except Exception as e:
        print("[webui] stt openai_compat error: %s" % e, flush=True)
        return ""


def _transcribe_deepgram(audio_path: str, suffix: str, p: dict) -> str:
    """Deepgram /v1/listen (raw body POST)."""
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        return ""
    base = (p.get("base_url") or "https://api.deepgram.com").rstrip("/")
    model = p.get("model") or "nova-2"
    params = p.get("params") or "smart_format=true&punctuate=true"
    url = "%s/v1/listen?model=%s&%s" % (base, model, params)
    MAX = 25 * 1024 * 1024
    with open(audio_path, "rb") as f:
        data = f.read()
    if len(data) > MAX:
        return ""
    mime = _mime_for(suffix)
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": "Token " + key,
        "Content-Type": mime,
        "Content-Length": str(len(data)),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print("[webui] stt deepgram non-200: %s" % resp.status, flush=True)
                return ""
            out = _json_mod.loads(resp.read().decode("utf-8", "replace"))
            ch = out.get("results", {}).get("channels", [])
            if ch and ch[0].get("alternatives"):
                return str(ch[0]["alternatives"][0].get("transcript") or "").strip()
            return ""
    except Exception as e:
        print("[webui] stt deepgram error: %s" % e, flush=True)
        return ""


def _transcribe_azure(audio_path: str, suffix: str, p: dict) -> str:
    """Azure Speech-to-Text REST (region + subscription key)."""
    key = os.environ.get(p.get("api_key_env", ""), "")
    region = os.environ.get(p.get("region_env", ""), "")
    if not key or not region:
        return ""
    fmt = {".wav": "wav", ".mp3": "mp3", ".ogg": "ogg", ".flac": "flac", ".m4a": "mp4", ".webm": "webm"}.get(suffix.lower(), "wav")
    url = "https://%s.stt.speech.microsoftazure.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed" % region
    with open(audio_path, "rb") as f:
        data = f.read()
    mime = "audio/%s" % fmt
    req = urllib.request.Request(url, data=data, headers={
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": mime,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return ""
            out = _json_mod.loads(resp.read().decode("utf-8", "replace"))
            return str(out.get("DisplayText") or "").strip()
    except Exception as e:
        print("[webui] stt azure error: %s" % e, flush=True)
        return ""


def _transcribe_assemblyai(audio_path: str, suffix: str, p: dict) -> str:
    """AssemblyAI: upload + submit + poll (token auth)."""
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        return ""
    base = (p.get("base_url") or "https://api.assemblyai.com").rstrip("/")
    with open(audio_path, "rb") as f:
        data = f.read()
    # upload
    try:
        up = urllib.request.Request(base + "/v2/upload", data=data, headers={"authorization": key, "content-type": _mime_for(suffix)})
        with urllib.request.urlopen(up, timeout=30) as r:
            upload_url = _json_mod.loads(r.read().decode()).get("upload_url", "")
        if not upload_url:
            return ""
        body = _json_mod.dumps({"audio_url": upload_url}).encode()
        st = urllib.request.Request(base + "/v2/transcript", data=body, headers={"authorization": key, "content-type": "application/json"})
        with urllib.request.urlopen(st, timeout=30) as r:
            tid = _json_mod.loads(r.read().decode()).get("id", "")
        # poll
        import time as _t
        for _ in range(30):
            with urllib.request.urlopen(urllib.request.Request(base + "/v2/transcript/%s" % tid, headers={"authorization": key})) as r:
                j = _json_mod.loads(r.read().decode())
            if j.get("status") == "completed":
                return str(j.get("text") or "").strip()
            if j.get("status") == "error":
                return ""
            _t.sleep(1)
    except Exception as e:
        print("[webui] stt assemblyai error: %s" % e, flush=True)
        return ""
    return ""


def _transcribe_elevenlabs(audio_path: str, suffix: str, p: dict) -> str:
    """ElevenLabs Speech-to-Text (scribe_v1)."""
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        return ""
    base = (p.get("base_url") or "https://api.elevenlabs.io").rstrip("/")
    model = p.get("model") or "scribe_v1"
    url = "%s/v1/speech-to-text?model_id=%s" % (base, model)
    with open(audio_path, "rb") as f:
        data = f.read()
    import io
    import uuid
    boundary = "----el%s" % uuid.uuid4().hex
    fname = "audio" + suffix
    crlf = b"\r\n"
    body = io.BytesIO()
    body.write(("--%s" % boundary).encode() + crlf)
    body.write(('Content-Disposition: form-data; name="file"; filename="%s"' % fname).encode() + crlf)
    body.write(("Content-Type: %s" % _mime_for(suffix)).encode() + crlf + crlf)
    body.write(data)
    body.write(crlf)
    body.write(("--%s--" % boundary).encode() + crlf)
    body_bytes = body.getvalue()
    req = urllib.request.Request(url, data=body_bytes, headers={
        "xi-api-key": key,
        "Content-Type": "multipart/form-data; boundary=%s" % boundary,
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                return ""
            out = _json_mod.loads(resp.read().decode("utf-8", "replace"))
            return str(out.get("text") or "").strip()
    except Exception as e:
        print("[webui] stt elevenlabs error: %s" % e, flush=True)
        return ""


def _transcribe_google(audio_path: str, suffix: str, p: dict) -> str:
    """Google Speech-to-Text v1 (sync recognize, key in query)."""
    key = os.environ.get(p.get("api_key_env", ""), "")
    if not key:
        return ""
    base = (p.get("base_url") or "https://speech.googleapis.com/v1").rstrip("/")
    url = "%s/speech:recognize?key=%s" % (base, key)
    with open(audio_path, "rb") as f:
        import base64
        b64 = base64.b64encode(f.read()).decode()
    gfmt = {".wav": "LINEAR16", ".flac": "FLAC", ".webm": "WEBM_OPUS", ".m4a": "MP3", ".mp3": "MP3", ".ogg": "OGG_OPUS"}.get(suffix.lower(), "WEBM_OPUS")
    payload = _json_mod.dumps({
        "config": {"encoding": gfmt, "sampleRateHertz": 16000, "languageCode": "en-US"},
        "audio": {"content": b64},
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return ""
            out = _json_mod.loads(resp.read().decode("utf-8", "replace"))
            parts = []
            for res in out.get("results", []):
                alts = res.get("alternatives", [])
                if alts:
                    parts.append(alts[0].get("transcript", ""))
            return " ".join(parts).strip()
    except Exception as e:
        print("[webui] stt google error: %s" % e, flush=True)
        return ""


_KIND_DISPATCH = {
    "openai_compat": _transcribe_openai_compat,
    "deepgram": _transcribe_deepgram,
    "azure": _transcribe_azure,
    "assemblyai": _transcribe_assemblyai,
    "elevenlabs": _transcribe_elevenlabs,
    "google": _transcribe_google,
}


def _transcribe_bytes(audio_bytes: bytes, suffix: str, prefer: str = "groq") -> str:
    """Run STT across all configured providers (priority order, prefer first).

    Tries the preferred provider first (if configured), then the rest in
    priority order. Returns the first non-empty transcript. Returns '' only
    if no provider is configured or all failed.
    """
    with tempfile.NamedTemporaryFile(prefix="webui-stt-", suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        provs = _enabled_providers()
        if not provs:
            return ""
        # re-order: preferred first if present
        ordered = sorted(provs, key=lambda x: 0 if x[0] == prefer else 1)
        for pid, p in ordered:
            kind = p.get("kind", "openai_compat")
            fn = _KIND_DISPATCH.get(kind)
            if not fn:
                continue
            try:
                text = fn(tmp_path, suffix, p)
                if text:
                    return text
            except Exception:
                continue
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

    # Transcribe the accumulated audio (last 2 chunks = ~3s for speed and
    # so each partial reflects the newest speech rather than re-processing a
    # long tail). Shorter window = partials update more responsively.
    combined = b"".join(buf["chunks"][-2:])
    if not _enabled_providers():
        return _json_response(handler, {"ok": False, "error": "no_stt_provider", "final": False}, 503)
    text = _transcribe_bytes(combined, suffix)
    if text:
        # Always advance the partial to the latest non-empty transcript.
        # A shorter-but-newer chunk (topic change, correction) must replace
        # a stale longer one — the client already dedupes identical partials.
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
    if not _enabled_providers():
        _clear_buffer(session)
        return _json_response(handler, {"ok": False, "error": "no_stt_provider", "final": True}, 503)
    text = _transcribe_bytes(combined, suffix, prefer="deepgram")
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


def handle_stt_providers(handler, parsed, method: str):
    """GET list / POST add / DELETE remove a custom STT provider.

    GET  -> { ok, providers: [{id,label,kind,configured,enabled}] }
    POST -> { id, label, kind, api_key_env, base_url, model, params?, region_env?, priority? }
           stores under "custom" in the config file (key never returned).
    DELETE -> { id } removes a custom provider.
    """
    import json as _json
    from urllib.parse import parse_qs as _pq

    if method == "GET":
        provs = _load_stt_providers()
        out = []
        for pid, p in provs.items():
            out.append({
                "id": pid,
                "label": p.get("label", pid),
                "kind": p.get("kind", "openai_compat"),
                "configured": bool(os.environ.get(p.get("api_key_env", ""), "")),
                "enabled": p.get("enabled", True),
                "custom": pid in _load_custom_ids(),
            })
        return _json_response(handler, {"ok": True, "providers": out})

    if method == "POST":
        try:
            length = int(handler.headers.get("Content-Length", 0) or 0)
            body = _json.loads(handler.rfile.read(length).decode("utf-8", "replace")) if length else {}
        except Exception as e:
            return _json_response(handler, {"error": str(e)}, 400)
        pid = str(body.get("id") or "").strip()
        if not pid or not re.match(r"^[a-z0-9_-]+$", pid):
            return _json_response(handler, {"error": "invalid id (lowercase alphanumeric/_/-)"}, 400)
        kind = body.get("kind", "openai_compat")
        if kind not in _KIND_DISPATCH:
            return _json_response(handler, {"error": "unknown kind"}, 400)
        entry = {
            "label": body.get("label") or pid,
            "kind": kind,
            "api_key_env": body.get("api_key_env") or ("%s_API_KEY" % pid.upper()),
            "base_url": body.get("base_url", ""),
            "model": body.get("model", ""),
            "priority": int(body.get("priority", 50)),
            "enabled": True,
        }
        if body.get("params"):
            entry["params"] = body["params"]
        if body.get("region_env"):
            entry["region_env"] = body["region_env"]
        # persist under "custom"
        try:
            with open(_STT_CONFIG_PATH) as f:
                cfg = _json.load(f)
        except Exception:
            cfg = {"providers": {}}
        cfg.setdefault("custom", {})[pid] = entry
        with open(_STT_CONFIG_PATH, "w") as f:
            _json.dump(cfg, f, indent=2)
        return _json_response(handler, {"ok": True, "id": pid})

    if method == "DELETE":
        qs = _pq(parsed.query)
        pid = (qs.get("id", [""])[0] or "").strip()
        try:
            with open(_STT_CONFIG_PATH) as f:
                cfg = _json.load(f)
            cfg.setdefault("custom", {}).pop(pid, None)
            with open(_STT_CONFIG_PATH, "w") as f:
                _json.dump(cfg, f, indent=2)
            return _json_response(handler, {"ok": True, "id": pid})
        except Exception as e:
            return _json_response(handler, {"error": str(e)}, 400)

    return _json_response(handler, {"error": "method not allowed"}, 405)
import re
