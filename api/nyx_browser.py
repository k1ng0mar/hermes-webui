"""Nyx shared browser — Playwright + system Chrome. Stills + tap/type.

GET  /api/nyx/browser/state
GET  /api/nyx/browser/frame.jpg
POST /api/nyx/browser/open   {url}
POST /api/nyx/browser/click  {x, y}   # 0..1 of the frame
POST /api/nyx/browser/type   {text, enter?}
POST /api/nyx/browser/key    {key}    # Enter, Backspace, Tab
POST /api/nyx/browser/scroll {dy}     # css pixels, +down
POST /api/nyx/browser/back
POST /api/nyx/browser/close
"""
from __future__ import annotations

import queue
import threading
import time

from api.helpers import bad, j

WIDTH, HEIGHT = 390, 844

_cmd: queue.Queue = queue.Queue()
_thread: threading.Thread | None = None
_ready = threading.Event()
_page_holder: dict = {}  # shared with the worker so snapshot can read DOM
_state = {"active": False, "url": "", "title": "", "width": WIDTH, "height": HEIGHT}
_frame = b""
_err: str | None = None
_last_poll = 0.0


def _worker() -> None:
    global _frame, _err, _state
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        _err = f"playwright missing: {e}"
        _ready.set()
        return

    try:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
            user_agent=(
                "Mozilla/5.0 (Linux; Android 12; Honor X6) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
            ),
        )
        page = context.new_page()
        _page_holder["page"] = page
    except Exception as e:
        _err = f"chrome launch failed: {e}"
        _ready.set()
        return

    _state["active"] = True
    _ready.set()

    def snap() -> None:
        global _frame
        _frame = page.screenshot(type="jpeg", quality=48)
        _state.update({"url": page.url or "", "title": page.title() or "", "active": True})

    while True:
        try:
            item = _cmd.get(timeout=0.28)
        except queue.Empty:
            if _last_poll and (time.time() - _last_poll) < 4:
                try:
                    snap()
                except Exception:
                    pass
            continue
        if item is None:
            break
        op, args, box, ev = item
        try:
            if op == "open":
                url = args[0]
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                time.sleep(0.2)
                snap()
                box["ok"] = dict(_state)
            elif op == "click":
                nx, ny = args
                page.mouse.click(nx * WIDTH, ny * HEIGHT)
                time.sleep(0.2)
                snap()
                box["ok"] = dict(_state)
            elif op == "type":
                text, enter = args
                if text:
                    page.keyboard.type(text, delay=12)
                if enter:
                    page.keyboard.press("Enter")
                time.sleep(0.2)
                snap()
                box["ok"] = dict(_state)
            elif op == "key":
                page.keyboard.press(args[0])
                time.sleep(0.15)
                snap()
                box["ok"] = dict(_state)
            elif op == "scroll":
                page.mouse.wheel(0, args[0])
                time.sleep(0.15)
                snap()
                box["ok"] = dict(_state)
            elif op == "back":
                page.go_back(wait_until="domcontentloaded", timeout=12000)
                time.sleep(0.2)
                snap()
                box["ok"] = dict(_state)
            elif op == "shot":
                snap()
                box["ok"] = dict(_state)
            elif op == "snapshot":
                # Read a structured digest of the current page so the agent can
                # resume from where the user took over. Bounded fields keep the
                # response small even on content-heavy pages.
                try:
                    snap()
                    snapshot = {
                        "url": _state.get("url", ""),
                        "title": _state.get("title", ""),
                        "headings": page.evaluate(
                            "() => Array.from(document.querySelectorAll('h1,h2,h3'))"
                            ".slice(0, 30).map(h => h.innerText.trim()).filter(Boolean)"
                        ),
                        "visible_text": page.evaluate(
                            "() => (document.body?.innerText || '')"
                            ".replace(/\\s+/g, ' ').trim().slice(0, 4000)"
                        ),
                        "form_fields": page.evaluate(
                            "() => Array.from(document.querySelectorAll('input,textarea,select'))"
                            ".slice(0, 30).map(el => ({"
                            "  tag: el.tagName.toLowerCase(),"
                            "  type: el.type || null,"
                            "  name: el.name || null,"
                            "  id: el.id || null,"
                            "  placeholder: el.placeholder || null,"
                            "  value: el.value || null"
                            "}))"
                        ),
                        "buttons": page.evaluate(
                            "() => Array.from(document.querySelectorAll('button, [role=button]'))"
                            ".slice(0, 20).map(b => b.innerText.trim()).filter(Boolean)"
                        ),
                    }
                    box["ok"] = snapshot
                except Exception as e:
                    box["err"] = f"snapshot failed: {e}"
            else:
                box["err"] = f"unknown op {op}"
        except Exception as e:
            box["err"] = str(e)
        ev.set()

    try:
        context.close()
        browser.close()
        pw.stop()
    except Exception:
        pass
    _state = {"active": False, "url": "", "title": "", "width": WIDTH, "height": HEIGHT}
    _frame = b""


def _ensure() -> None:
    global _thread, _err
    if _thread and _thread.is_alive():
        return
    _err = None
    _ready.clear()
    _thread = threading.Thread(target=_worker, name="nyx-browser", daemon=True)
    _thread.start()
    if not _ready.wait(20):
        raise RuntimeError("browser worker did not start")
    if _err:
        raise RuntimeError(_err)


def _call(op: str, *args, timeout: float = 25.0) -> dict:
    _ensure()
    ev = threading.Event()
    box: dict = {}
    _cmd.put((op, args, box, ev))
    if not ev.wait(timeout):
        raise TimeoutError(op)
    if box.get("err"):
        raise RuntimeError(box["err"])
    return box.get("ok") or dict(_state)


def handle_state(handler):
    return j(handler, dict(_state))


def handle_frame(handler):
    global _last_poll
    _last_poll = time.time()
    try:
        if not (_thread and _thread.is_alive() and _state.get("active")):
            _call("shot")
        elif not _frame:
            _call("shot")
        data = _frame
    except Exception as e:
        return bad(handler, str(e), 503)
    if not data:
        return bad(handler, "empty frame", 503)
    handler.send_response(200)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)
    return True


def handle_open(handler, body):
    url = str((body or {}).get("url") or "").strip()
    if not url:
        return bad(handler, "url required")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        st = _call("open", url)
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)


def handle_click(handler, body):
    try:
        nx = min(1.0, max(0.0, float((body or {}).get("x"))))
        ny = min(1.0, max(0.0, float((body or {}).get("y"))))
    except (TypeError, ValueError):
        return bad(handler, "x,y required (0..1)")
    try:
        st = _call("click", nx, ny)
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)


def handle_type(handler, body):
    text = str((body or {}).get("text") or "")
    enter = bool((body or {}).get("enter"))
    if not text and not enter:
        return bad(handler, "text required")
    try:
        st = _call("type", text, enter)
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)


def handle_key(handler, body):
    key = str((body or {}).get("key") or "").strip()
    if not key:
        return bad(handler, "key required")
    try:
        st = _call("key", key)
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)


def handle_scroll(handler, body):
    try:
        dy = float((body or {}).get("dy") or 0)
    except (TypeError, ValueError):
        return bad(handler, "dy required")
    try:
        st = _call("scroll", dy)
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)


def handle_back(handler, body):
    try:
        st = _call("back")
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)


def handle_close(handler, body):
    global _thread
    if _thread and _thread.is_alive():
        _cmd.put(None)
        _thread.join(timeout=8)
    _thread = None
    return j(handler, {"ok": True, "active": False})


def handle_snapshot(handler, body):
    """Read a structured page digest: url, title, headings, visible text (capped),
    form fields, button labels. Use this when the user gives back control after
    a take-over so the agent resumes with the user's final-page context, not
    just the URL it last acted on."""
    try:
        st = _call("snapshot")
        return j(handler, {"ok": True, **st})
    except Exception as e:
        return bad(handler, str(e), 503)
