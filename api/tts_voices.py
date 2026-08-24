"""TTS voice list endpoint.

GET /api/tts/voices?engine=edge
GET /api/tts/voices?engine=openai&base_url=...&key=...

Returns { voices: [{id, name, locale?, gender?}] }
"""

import json
import urllib.request
import urllib.error

# Microsoft Edge TTS neural voices — stable set, no API call needed
_EDGE_VOICES = [
    {"id": "en-US-AriaNeural", "name": "Aria (US female)", "locale": "en-US", "gender": "female"},
    {"id": "en-US-GuyNeural", "name": "Guy (US male)", "locale": "en-US", "gender": "male"},
    {"id": "en-US-JennyNeural", "name": "Jenny (US female)", "locale": "en-US", "gender": "female"},
    {"id": "en-US-TonyNeural", "name": "Tony (US male)", "locale": "en-US", "gender": "male"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia (GB female)", "locale": "en-GB", "gender": "female"},
    {"id": "en-GB-RyanNeural", "name": "Ryan (GB male)", "locale": "en-GB", "gender": "male"},
    {"id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao (CN female)", "locale": "zh-CN", "gender": "female"},
    {"id": "zh-CN-YunxiNeural", "name": "Yunxi (CN male)", "locale": "zh-CN", "gender": "male"},
    {"id": "fr-CA-SylvieNeural", "name": "Sylvie (CA female)", "locale": "fr-CA", "gender": "female"},
    {"id": "fr-FR-DeniseNeural", "name": "Denise (FR female)", "locale": "fr-FR", "gender": "female"},
    {"id": "ja-JP-NanamiNeural", "name": "Nanami (JP female)", "locale": "ja-JP", "gender": "female"},
    {"id": "ko-KR-SunHiNeural", "name": "SunHi (KR female)", "locale": "ko-KR", "gender": "female"},
    {"id": "pt-BR-FranciscaNeural", "name": "Francisca (BR female)", "locale": "pt-BR", "gender": "female"},
    {"id": "es-ES-ElviraNeural", "name": "Elvira (ES female)", "locale": "es-ES", "gender": "female"},
    {"id": "de-DE-KatjaNeural", "name": "Katja (DE female)", "locale": "de-DE", "gender": "female"},
    {"id": "ar-SA-ZariyahNeural", "name": "Zariyah (SA female)", "locale": "ar-SA", "gender": "female"},
]

_OPENAI_VOICES = [
    {"id": "alloy", "name": "Alloy"},
    {"id": "echo", "name": "Echo"},
    {"id": "fable", "name": "Fable"},
    {"id": "onyx", "name": "Onyx"},
    {"id": "nova", "name": "Nova"},
    {"id": "shimmer", "name": "Shimmer"},
]

# Deepgram Aura voices — fixed set
_DEEPGRAM_VOICES = [
    {"id": "asteria", "name": "Asteria (female)"},
    {"id": "luna", "name": "Luna (female)"},
    {"id": "stella", "name": "Stella (female)"},
    {"id": "athena", "name": "Athena (female)"},
    {"id": "hera", "name": "Hera (female)"},
    {"id": "orion", "name": "Orion (male)"},
    {"id": "arcas", "name": "Arcas (male)"},
    {"id": "perseus", "name": "Perseus (male)"},
    {"id": "angus", "name": "Angus (male)"},
    {"id": "orpheus", "name": "Orpheus (male)"},
    {"id": "helios", "name": "Helios (male)"},
    {"id": "zeus", "name": "Zeus (male)"},
]


def j(handler, obj, status=200):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps(obj).encode())


def bad(handler, msg, status=400):
    return j(handler, {"error": msg}, status)


def handle_tts_voices(handler, parsed):
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    engine = (qs.get("engine", ["edge"])[0] or "edge").strip().lower()
    base_url = (qs.get("base_url", [""])[0] or "").strip()
    key = (qs.get("key", [""])[0] or "").strip()

    if engine == "edge":
        return j(handler, {"voices": _EDGE_VOICES})

    if engine == "openai":
        return j(handler, {"voices": _OPENAI_VOICES})

    if engine == "deepgram":
        return j(handler, {"voices": _DEEPGRAM_VOICES})

    if engine == "fish":
        # Fish Audio reference voices — fetch from API if key present
        if not key:
            return j(handler, {"voices": []})
        try:
            req = urllib.request.Request("https://api.fish.audio/v1/voices")
            req.add_header("Authorization", f"Bearer {key}")
            req.add_header("Accept", "application/json")
            resp = urllib.request.urlopen(req, timeout=10)
            body = json.loads(resp.read())
            raw = body.get("data") if isinstance(body, dict) else body
            if not isinstance(raw, list):
                raw = []
            voices = []
            for v in raw:
                if isinstance(v, dict):
                    vid = v.get("id") or v.get("voice_id") or v.get("name")
                    if vid:
                        voices.append({
                            "id": str(vid),
                            "name": str(v.get("name") or v.get("title") or vid),
                        })
            return j(handler, {"voices": voices})
        except Exception as e:
            return bad(handler, f"fish voice list failed: {e}", 502)

    # Custom / generic OpenAI-compatible — try to list voices from the endpoint
    if engine == "custom" and base_url:
        clean = base_url.rstrip("/").replace("/v1", "")
        voice_url = clean + "/v1/audio/voices"
        req = urllib.request.Request(voice_url)
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Accept", "application/json")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            body = json.loads(resp.read())
            # Normalize whatever shape the API returns
            raw = body
            if isinstance(raw, dict):
                raw = raw.get("data") or raw.get("voices") or raw.get("voices") or []
            if not isinstance(raw, list):
                raw = [raw]
            voices = []
            for v in raw:
                if isinstance(v, str):
                    voices.append({"id": v, "name": v})
                elif isinstance(v, dict):
                    vid = v.get("id") or v.get("voice_id") or v.get("name")
                    if vid:
                        voices.append({
                            "id": str(vid),
                            "name": str(v.get("name") or v.get("voice_id") or vid),
                            "locale": v.get("locale") or v.get("language"),
                        })
            return j(handler, {"voices": voices})
        except urllib.error.HTTPError as e:
            return bad(handler, f"voice list request failed: {e.code}", 502)
        except Exception as e:
            return bad(handler, f"voice list request failed: {e}", 502)

    return bad(handler, f"unknown engine '{engine}' or missing base_url for custom", 400)