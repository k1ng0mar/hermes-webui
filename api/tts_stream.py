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

    # Voice allowlist — same as /api/tts (defense against SSRF/path traversal)
    try:
        from api.routes import _TTS_ALLOWED_VOICES as allowed
    except Exception:
        allowed = None
    if allowed is not None and voice not in allowed:
        from api.helpers import bad as _bad
        return _bad(handler, "invalid voice", 400)

    try:
        import edge_tts
    except ImportError:
        from api.helpers import bad as _bad
        return _bad(handler, "Edge TTS engine not installed", 503)

    kwargs = {}
    if rate_str:
        kwargs["rate"] = rate_str
    if pitch_str:
        kwargs["pitch"] = pitch_str

    try:
        comm = edge_tts.Communicate(text, voice, **kwargs)
        # Explicit HTTP/1.1 chunked transfer-encoding — each audio chunk is
        # framed as its own chunk, so the client gets data immediately and the
        # connection closes cleanly when we send the terminating 0-chunk.
        handler.send_response(200)
        handler.send_header("Content-Type", "audio/mpeg")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Accel-Buffering", "no")
        handler.end_headers()

        def _write_chunk(data: bytes):
            handler.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
            handler.wfile.flush()

        sent = 0
        for chunk in comm.stream_sync():
            if chunk.get("type") == "audio" and chunk.get("data"):
                data = chunk["data"]
                try:
                    _write_chunk(data)
                    sent += len(data)
                except (BrokenPipeError, ConnectionResetError):
                    break  # client hung up — stop streaming
        # Terminating chunk — signals end of stream and closes cleanly.
        try:
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        except Exception:
            pass
        if sent == 0:
            pass  # empty synthesis — client sees an empty chunked body
    except Exception:
        import traceback as _tb
        print("[webui] tts-stream error: " + _tb.format_exc(), flush=True)
        try:
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        except Exception:
            pass
    return True
