import os
import time
import hmac
import socket
import secrets
import threading
from datetime import datetime
from contextlib import asynccontextmanager

import cv2
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

# ================= SETTINGS =================
PORT = 8000
SESSION_HOURS = 12        # how long a login stays valid
MAX_FAILS = 5             # wrong keys allowed before lockout
LOCKOUT_SECONDS = 300     # 5 minute lockout
KEY_FILE = "access_key.txt"
# ============================================


def load_key():
    """Read the access key from a file, or make a new random one the first time."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            return f.read().strip()
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    key = "".join(secrets.choice(alphabet) for _ in range(8))
    with open(KEY_FILE, "w") as f:
        f.write(key)
    return key


ACCESS_KEY = load_key()

lock = threading.Lock()
stop_event = threading.Event()
latest_jpeg = None
people_count = 0
sessions = {}     # session token -> expiry time
fails = {}        # IP address -> (wrong attempts, locked until)


# ---------------- Camera + AI thread ----------------
def camera_loop():
    global latest_jpeg, people_count
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        results = model.track(frame, classes=[0], conf=0.5, persist=True,
                              tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes
        count = 0
        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(tid)}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, stamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()
                people_count = count

    cap.release()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=camera_loop, daemon=True).start()
    yield
    stop_event.set()


app = FastAPI(lifespan=lifespan)


# ---------------- Login helpers ----------------
def client_ip(request: Request):
    return request.client.host if request.client else "unknown"


def is_logged_in(request: Request):
    token = request.cookies.get("session")
    expiry = sessions.get(token)
    if expiry and expiry > time.time():
        return True
    sessions.pop(token, None)
    return False


@app.post("/api/login")
async def login(request: Request):
    ip = client_ip(request)
    count, locked_until = fails.get(ip, (0, 0))
    now = time.time()

    if locked_until > now:
        wait = int(locked_until - now)
        return JSONResponse({"ok": False, "error": f"Too many attempts. Try again in {wait}s."},
                            status_code=429)

    try:
        data = await request.json()
    except Exception:
        data = {}
    key = str(data.get("key", "")).strip()

    # compare_digest avoids timing leaks when checking the key
    if hmac.compare_digest(key.encode(), ACCESS_KEY.encode()):
        fails.pop(ip, None)
        token = secrets.token_urlsafe(32)
        sessions[token] = now + SESSION_HOURS * 3600
        resp = JSONResponse({"ok": True})
        resp.set_cookie("session", token, httponly=True, samesite="lax",
                        max_age=SESSION_HOURS * 3600)
        return resp

    count += 1
    if count >= MAX_FAILS:
        fails[ip] = (0, now + LOCKOUT_SECONDS)
        return JSONResponse({"ok": False, "error": "Too many attempts. Locked for 5 minutes."},
                            status_code=429)
    fails[ip] = (count, 0)
    return JSONResponse({"ok": False, "error": f"Wrong key. {MAX_FAILS - count} attempts left."},
                        status_code=401)


@app.post("/api/logout")
async def logout(request: Request):
    sessions.pop(request.cookies.get("session"), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp


# ---------------- Protected routes ----------------
def mjpeg_generator():
    while not stop_event.is_set():
        with lock:
            frame = latest_jpeg
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.get("/video")
def video(request: Request):
    if not is_logged_in(request):
        return Response(status_code=401)
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot")
def snapshot(request: Request):
    if not is_logged_in(request):
        return Response(status_code=401)
    with lock:
        frame = latest_jpeg
    if frame is None:
        return Response(status_code=503)
    name = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
    return Response(frame, media_type="image/jpeg",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/status")
def status(request: Request):
    if not is_logged_in(request):
        return JSONResponse({"error": "login required"}, status_code=401)
    with lock:
        c = people_count
    return {"people": c, "time": datetime.now().strftime("%H:%M:%S")}


# ---------------- Web pages ----------------
STYLE = """
<style>
:root{--bg:#f4f6f8;--card:#ffffff;--text:#1a202c;--muted:#64748b;--accent:#0fa3b1;--border:#d8dee6}
[data-theme=dark]{--bg:#0b1220;--card:#131c2e;--text:#e6edf6;--muted:#93a4bd;--border:#243049}
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;
  background:var(--card);border-bottom:1px solid var(--border)}
h1{font-size:18px;margin:0;color:var(--accent)}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;
  font-size:14px;cursor:pointer;margin-left:6px}
button.ghost{background:transparent;color:var(--text);border:1px solid var(--border)}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.wrap{max-width:900px;margin:18px auto;padding:0 14px}
img.video{width:100%;border-radius:10px;background:#000;min-height:200px}
.row{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
.stat{flex:1;min-width:140px;text-align:center}
.stat b{display:block;font-size:26px;color:var(--accent)}
.stat span{color:var(--muted);font-size:13px}
input{width:100%;padding:11px;border-radius:8px;border:1px solid var(--border);
  background:var(--bg);color:var(--text);font-size:16px;margin:10px 0}
.err{color:#d64545;min-height:20px;font-size:14px}
.login{max-width:360px;margin:12vh auto;text-align:center}
</style>
"""

THEME_JS = """
<script>
function applyTheme(t){document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);}
applyTheme(localStorage.getItem('theme') ||
  (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
function toggleTheme(){applyTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');}
</script>
"""

LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SentinelEye - Login</title>__STYLE____THEME__</head><body>
<div class="login card">
  <h1>SentinelEye</h1>
  <p style="color:var(--muted)">Enter the access key shown on the camera device.</p>
  <input id="key" type="password" placeholder="Access key" autofocus>
  <div class="err" id="msg"></div>
  <button onclick="login()" style="width:100%;margin:0">Unlock</button>
  <p><button class="ghost" onclick="toggleTheme()">Dark / Light</button></p>
</div>
<script>
document.getElementById('key').addEventListener('keydown', e => { if(e.key==='Enter') login(); });
async function login(){
  const key = document.getElementById('key').value;
  const r = await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key})});
  const d = await r.json();
  if(d.ok){ location.reload(); } else { document.getElementById('msg').textContent = d.error; }
}
</script></body></html>""".replace("__STYLE__", STYLE).replace("__THEME__", THEME_JS)

DASH_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SentinelEye - Live</title>__STYLE____THEME__</head><body>
<header>
  <h1>SentinelEye Live</h1>
  <div>
    <button class="ghost" onclick="toggleTheme()">Dark / Light</button>
    <button class="ghost" onclick="logout()">Logout</button>
  </div>
</header>
<div class="wrap">
  <div class="card">
    <img id="vid" class="video" src="/video" alt="Live video (starting...)">
    <div class="row">
      <div class="card stat"><b id="people">0</b><span>People in view</span></div>
      <div class="card stat"><b id="clock">--:--</b><span>Server time</span></div>
      <div class="card stat">
        <button onclick="location.href='/snapshot'">Snapshot</button>
        <button onclick="document.getElementById('vid').requestFullscreen()">Fullscreen</button>
      </div>
    </div>
  </div>
</div>
<script>
async function poll(){
  try{
    const r = await fetch('/api/status');
    if(r.status===401){ location.reload(); return; }
    const d = await r.json();
    document.getElementById('people').textContent = d.people;
    document.getElementById('clock').textContent = d.time;
  }catch(e){}
}
setInterval(poll, 1000); poll();
async function logout(){ await fetch('/api/logout',{method:'POST'}); location.reload(); }
</script></body></html>""".replace("__STYLE__", STYLE).replace("__THEME__", THEME_JS)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return DASH_HTML if is_logged_in(request) else LOGIN_HTML


# ---------------- Start ----------------
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ip = lan_ip()
    print("=" * 50)
    print(" SentinelEye server starting")
    print(f" On this laptop : http://localhost:{PORT}")
    print(f" On other devices (same Wi-Fi): http://{ip}:{PORT}")
    print(f" ACCESS KEY     : {ACCESS_KEY}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=PORT)