def _handle_tts_stream(handler, parsed):
    """Streaming TTS: yields mp3 chunks as edge_tts generates them.

    Unlike /api/tts (which buffers the whole file so Content-Length is known),
    this endpoint streams each audio chunk immediately with explicit HTTP/1.1
    chunked transfer-encoding. The client can start playback as soon as the
    first chunk arrives, cutting perceived latency from ~2-4s to ~200-400ms.

    GET /api/voice/tts-stream?text=...&voice=...&rate=...&pitch=...
    """
    from urllib.parse import parse_qs as _parse_qs
    qs = _parse_qs(parsed.query)
    text = (qs.get("text", [""])[0] or "").strip()
    if not text:
        from api.helpers import bad as _bad
        return _bad(handler, "text is required", 400)
    if len(text) > 5000:
        from api.helpers import bad as _bad
        return _bad(handler, "text too long (max 5000 characters)", 400)

    voice = qs.get("voice", ["en-US-JennyNeural"])[0]
    rate_str = qs.get("rate", [""])[0]
    pitch_str = qs.get("pitch", [""])[0]
    engine = qs.get("engine", ["edge"])[0]
    api_key = qs.get("key", [""])[0]
    base_url = qs.get("base_url", [""])[0]

    # Voice allowlist — same as /api/tts (defense against SSRF/path traversal)
    try:
        from api.routes import _TTS_ALLOWED_VOICES as allowed
    except Exception:
        allowed = None
    if allowed is not None and voice not in allowed and engine == "edge":
        from api.helpers import bad as _bad
        return _bad(handler, "invalid voice", 400)

    try:
        import edge_tts
    except ImportError:
        from api.helpers import bad as _bad
        return _bad(handler, "Edge TTS engine not installed", 503)

    # ── Provider dispatch ──
    # edge (default) — local streaming via edge_tts
    if engine == "edge":
        _handle_edge_stream(handler, text, voice, rate_str, pitch_str)
        return True
    # deepgram — POST /v1/speak with api key
    if engine == "deepgram":
        _handle_deepgram_stream(handler, text, voice, api_key)
        return True
    # fish audio — POST /v1/tts with api key
    if engine == "fish":
        _handle_fish_stream(handler, text, voice, api_key)
        return True
    # openai-compatible (openai, custom)
    if engine in ("openai", "custom"):
        _handle_openai_stream(handler, text, voice, api_key, base_url)
        return True

    from api.helpers import bad as _bad
    return _bad(handler, f"unknown engine: {engine}", 400)


def _send_chunked_headers(handler):
    """Send 200 + chunked transfer-encoding headers."""
    handler.send_response(200)
    handler.send_header("Content-Type", "audio/mpeg")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()


def _write_chunk(handler, data: bytes):
    handler.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
    handler.wfile.flush()


def _finish_chunked(handler):
    try:
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    except Exception:
        pass


def _handle_edge_stream(handler, text, voice, rate_str, pitch_str):
    """Stream edge_tts output chunk by chunk."""
    import edge_tts

    kwargs = {}
    if rate_str:
        kwargs["rate"] = rate_str
    if pitch_str:
        kwargs["pitch"] = pitch_str
    try:
        comm = edge_tts.Communicate(text, voice, **kwargs)
        _send_chunked_headers(handler)
        for chunk in comm.stream_sync():
            if chunk.get("type") == "audio" and chunk.get("data"):
                try:
                    _write_chunk(handler, chunk["data"])
                except (BrokenPipeError, ConnectionResetError):
                    break
        _finish_chunked(handler)
    except Exception:
        import traceback as _tb
        print("[webui] tts-stream edge error: " + _tb.format_exc(), flush=True)
        _finish_chunked(handler)


def _handle_deepgram_stream(handler, text, voice, api_key):
    """Stream Deepgram TTS (POST /v1/speak, Authorization: Token)."""
    import urllib.request as _urlreq

    if not api_key:
        from api.helpers import bad as _bad
        return _bad(handler, "Deepgram API key required", 400)
    try:
        url = f"https://api.deepgram.com/v1/speak?model=aura-2&voice={voice}"
        payload = {"text": text}
        req = _urlreq.Request(
            url,
            data=__import__("json").dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        _send_chunked_headers(handler)
        with _urlreq.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                try:
                    _write_chunk(handler, chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
        _finish_chunked(handler)
    except Exception:
        import traceback as _tb
        print("[webui] tts-stream deepgram error: " + _tb.format_exc(), flush=True)
        _finish_chunked(handler)


def _handle_fish_stream(handler, text, voice, api_key):
    """Stream Fish Audio TTS (POST /v1/tts, Authorization: Bearer)."""
    import urllib.request as _urlreq

    if not api_key:
        from api.helpers import bad as _bad
        return _bad(handler, "Fish Audio API key required", 400)
    try:
        url = "https://api.fish.audio/v1/tts"
        payload = {"text": text, "reference_id": voice or "default", "format": "mp3"}
        req = _urlreq.Request(
            url,
            data=__import__("json").dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        _send_chunked_headers(handler)
        with _urlreq.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                try:
                    _write_chunk(handler, chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
        _finish_chunked(handler)
    except Exception:
        import traceback as _tb
        print("[webui] tts-stream fish error: " + _tb.format_exc(), flush=True)
        _finish_chunked(handler)


def _handle_openai_stream(handler, text, voice, api_key, base_url):
    """Stream OpenAI-compatible TTS (POST {base}/v1/audio/speech)."""
    import urllib.request as _urlreq
    from urllib.parse import urlunsplit as _urlunsplit

    if not api_key:
        from api.helpers import bad as _bad
        return _bad(handler, "API key required", 400)
    try:
        base = base_url or _urlunsplit(("https", "api.openai.com", "/v1", "", ""))
        url = base.rstrip("/") + "/audio/speech"
        payload = {"model": "gpt-4o-mini-tts", "voice": voice or "alloy", "input": text}
        req = _urlreq.Request(
            url,
            data=__import__("json").dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        _send_chunked_headers(handler)
        with _urlreq.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                try:
                    _write_chunk(handler, chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
        _finish_chunked(handler)
    except Exception:
        import traceback as _tb
        print("[webui] tts-stream openai error: " + _tb.format_exc(), flush=True)
        _finish_chunked(handler)
