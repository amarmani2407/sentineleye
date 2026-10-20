import os
import re
import math
import time
import hmac
import socket
import sqlite3
import secrets
import threading
import urllib.parse
import urllib.request
from collections import deque
from contextlib import asynccontextmanager, closing
from datetime import datetime

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from ultralytics import YOLO

# ======================= SETTINGS (change these) =======================
PORT = 8000
CAMERA_INDEX = 0

# Login
SESSION_HOURS = 12
MAX_FAILS = 5
LOCKOUT_SECONDS = 300
KEY_FILE = "access_key.txt"

# Rules
# Restricted zone as fractions of the screen (x, y). Default = right side.
ZONE_FRACTIONS = [(0.55, 0.15), (0.95, 0.15), (0.95, 0.95), (0.55, 0.95)]
LOITER_SECONDS = 15          # staying in one spot this long = loitering
LOITER_RADIUS = 60           # pixels a person may move and still count as "staying"
FALL_ANGLE = 55              # torso tilt from vertical (degrees) that counts as lying
FALL_HOLD_SECONDS = 1.0      # must stay lying this long to count as a fall
KP_CONF = 0.5                # minimum confidence to trust a body point
AFTER_HOURS_START = 22       # 10 PM
AFTER_HOURS_END = 6          # 6 AM

# Alerts: seconds before the same person triggers the same alert type again
COOLDOWNS = {"zone": 15, "loiter": 30, "fall": 20, "after": 60}

# Evidence
CLIP_PRE_SECONDS = 4         # seconds of video saved from BEFORE the alert
CLIP_POST_SECONDS = 6        # seconds saved AFTER the alert
MAX_EVIDENCE_MB = 500        # oldest files are deleted when this is exceeded
# =======================================================================

EVIDENCE_DIR = "evidence"
SNAP_DIR = os.path.join(EVIDENCE_DIR, "snapshots")
CLIP_DIR = os.path.join(EVIDENCE_DIR, "clips")
DB_FILE = "events.db"
TELEGRAM_FILE = "telegram.txt"   # optional: line 1 = bot token, line 2 = chat id

os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)

# ---------------- Shared state ----------------
lock = threading.Lock()
stop_event = threading.Event()
latest_jpeg = None
people_count = 0
alert_now = False
recording_name = None
sim_after_hours = False
sessions = {}        # session token -> expiry time
fails = {}           # IP address -> (wrong attempts, locked until)


# ---------------- Access key ----------------
def load_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            return f.read().strip()
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    key = "".join(secrets.choice(alphabet) for _ in range(8))
    with open(KEY_FILE, "w") as f:
        f.write(key)
    return key


ACCESS_KEY = load_key()


# ---------------- Database (SQLite event log) ----------------
db_lock = threading.Lock()


def db_connect():
    return sqlite3.connect(DB_FILE)


def init_db():
    with db_lock, closing(db_connect()) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            person_id INTEGER,
            message TEXT NOT NULL,
            snapshot TEXT,
            clip TEXT)""")
        con.commit()


def add_event(kind, tid, message, snapshot, clip):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_lock, closing(db_connect()) as con:
        con.execute(
            "INSERT INTO events (ts, kind, person_id, message, snapshot, clip) VALUES (?,?,?,?,?,?)",
            (ts, kind, tid, message, snapshot, clip))
        con.commit()


def get_events(limit=40):
    with db_lock, closing(db_connect()) as con:
        rows = con.execute(
            "SELECT id, ts, kind, person_id, message, snapshot, clip "
            "FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    with lock:
        busy = recording_name
    out = []
    for id_, ts, kind, pid, msg, snap, clip in rows:
        # Evidence may have been auto-deleted to free space
        if snap and not os.path.exists(os.path.join(SNAP_DIR, snap)):
            snap = None
        if clip and not os.path.exists(os.path.join(CLIP_DIR, clip)):
            clip = None
        out.append({"id": id_, "ts": ts, "kind": kind, "person": pid, "message": msg,
                    "snapshot": snap, "clip": clip, "recording": bool(clip and clip == busy)})
    return out


def count_today():
    today = datetime.now().strftime("%Y-%m-%d")
    with db_lock, closing(db_connect()) as con:
        return con.execute("SELECT COUNT(*) FROM events WHERE ts LIKE ?", (today + "%",)).fetchone()[0]


def cleanup_evidence():
    """Loop storage: delete the oldest evidence files when the folder gets too big."""
    files = []
    for folder in (SNAP_DIR, CLIP_DIR):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                files.append((os.path.getmtime(path), os.path.getsize(path), path))
    total = sum(size for _, size, _ in files)
    limit = MAX_EVIDENCE_MB * 1024 * 1024
    with lock:
        busy = recording_name
    for _, size, path in sorted(files):
        if total <= limit:
            break
        if busy and os.path.basename(path) == busy:
            continue
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass


# ---------------- Optional Telegram alerts ----------------
def load_telegram():
    if not os.path.exists(TELEGRAM_FILE):
        return None
    with open(TELEGRAM_FILE) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return (lines[0], lines[1]) if len(lines) >= 2 else None


TELEGRAM = load_telegram()


def send_telegram(text):
    if not TELEGRAM:
        return

    def _send():
        try:
            token, chat = TELEGRAM
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
            urllib.request.urlopen(url, data=data, timeout=10).read()
        except Exception as e:
            print("Telegram message failed:", type(e).__name__)

    threading.Thread(target=_send, daemon=True).start()


# ---------------- Rule helpers ----------------
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12   # body point numbers


def midpoint(kps, confs, a, b):
    if confs is not None and (confs[a] < KP_CONF or confs[b] < KP_CONF):
        return None
    return ((kps[a][0] + kps[b][0]) / 2, (kps[a][1] + kps[b][1]) / 2)


def torso_angle(kps, confs):
    """0 = upright, 90 = lying flat. None if shoulders/hips are not visible."""
    sh = midpoint(kps, confs, L_SH, R_SH)
    hip = midpoint(kps, confs, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None
    return math.degrees(math.atan2(abs(sh[0] - hip[0]), abs(sh[1] - hip[1])))


def is_after_hours():
    if sim_after_hours:
        return True
    hour = datetime.now().hour
    return hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END


def make_writer(path, fps, size):
    """Try H.264 first (plays in browsers), fall back to mp4v."""
    for code in ("avc1", "mp4v"):
        try:
            if hasattr(cv2, "VideoWriter_fourcc"):
                fourcc = cv2.VideoWriter_fourcc(*code)
            else:
                fourcc = cv2.VideoWriter.fourcc(*code)
            writer = cv2.VideoWriter(path, fourcc, fps, size)
            if writer.isOpened():
                return writer
            writer.release()
        except Exception:
            pass
    return None


# ---------------- Camera + AI thread ----------------
def camera_loop():
    global latest_jpeg, people_count, alert_now, recording_name

    model = YOLO("yolov8n-pose.pt")     # finds people AND body points
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open the webcam. Close other apps using it and restart.")
        return

    anchors = {}          # person ID -> (x, y, time they started staying there)
    lying_since = {}      # person ID -> time they started lying
    last_alert = {}       # (person ID, kind) -> last time the alert fired
    pre_buffer = deque()  # (time, frame) kept for the "before" part of clips
    clip = None           # the clip being recorded right now
    fps_est = 10.0
    last_t = time.time()

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt)

        h, w = frame.shape[:2]
        zone = np.array([(int(x * w), int(y * h)) for x, y in ZONE_FRACTIONS], np.int32)

        r = model.track(frame, conf=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        view = r.plot(boxes=False)    # draws the skeleton

        # Draw the restricted zone
        overlay = view.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
        view = cv2.addWeighted(overlay, 0.2, view, 0.8, 0)
        cv2.polylines(view, [zone], True, (0, 0, 255), 2)
        cv2.putText(view, "RESTRICTED ZONE", (int(zone[0][0]) + 5, int(zone[0][1]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        candidates = []   # (kind, person id, message) found in this frame
        seen = set()

        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            kp_xy = r.keypoints.xy.cpu().numpy()
            kp_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

            for i, (box, tid) in enumerate(zip(boxes, ids)):
                x1, y1, x2, y2 = map(int, box)
                seen.add(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                flags = []

                # Rule 1: restricted zone (uses the feet = bottom centre of the box)
                if cv2.pointPolygonTest(zone, (float(cx), float(y2)), False) >= 0:
                    flags.append("IN ZONE")
                    candidates.append(("zone", tid, f"Person {tid} entered the restricted zone"))

                # Rule 2: loitering
                ax, ay, start = anchors.get(tid, (cx, cy, now))
                if math.hypot(cx - ax, cy - ay) > LOITER_RADIUS:
                    ax, ay, start = cx, cy, now
                anchors[tid] = (ax, ay, start)
                stayed = now - start
                if stayed >= LOITER_SECONDS:
                    flags.append(f"LOITERING {int(stayed)}s")
                    candidates.append(("loiter", tid, f"Person {tid} stayed in one spot for {int(stayed)} seconds"))

                # Rule 3: after-hours
                if is_after_hours():
                    flags.append("AFTER-HOURS")
                    candidates.append(("after", tid, f"Person {tid} detected during quiet hours"))

                # Rule 4: fall
                angle = torso_angle(kp_xy[i], kp_conf[i] if kp_conf is not None else None)
                lying = (angle > FALL_ANGLE) if angle is not None else ((x2 - x1) > 1.2 * (y2 - y1))
                if lying:
                    lying_since.setdefault(tid, now)
                else:
                    lying_since.pop(tid, None)
                if tid in lying_since and now - lying_since[tid] >= FALL_HOLD_SECONDS:
                    flags.append("FALL")
                    candidates.append(("fall", tid, f"Person {tid} appears to have fallen"))

                color = (0, 0, 255) if flags else (0, 255, 0)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tid}" + (" " + ", ".join(flags) if flags else "")
                cv2.putText(view, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Forget people who left the frame
        for state in (anchors, lying_since):
            for k in list(state):
                if k not in seen:
                    del state[k]

        # Banner and timestamp
        if candidates:
            kinds = sorted({k.upper() for k, _, _ in candidates})
            cv2.rectangle(view, (0, 0), (w, 36), (0, 0, 255), -1)
            cv2.putText(view, "ALERT: " + ", ".join(kinds), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Apply cooldowns so we don't flood the log
        fired = []
        for kind, tid, msg in candidates:
            key = (tid, kind)
            if now - last_alert.get(key, 0) >= COOLDOWNS[kind]:
                last_alert[key] = now
                fired.append((kind, tid, msg))

        if fired:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_name = f"snap_{stamp}.jpg"
            cv2.imwrite(os.path.join(SNAP_DIR, snap_name), view)

            # Start a clip (or extend the one already recording)
            if clip is None:
                clip_name = f"clip_{stamp}.mp4"
                writer = make_writer(os.path.join(CLIP_DIR, clip_name),
                                     max(5.0, min(25.0, fps_est)), (w, h))
                if writer is not None:
                    for _, old in pre_buffer:
                        writer.write(old)
                    clip = {"writer": writer, "name": clip_name, "end": now + CLIP_POST_SECONDS}
                    with lock:
                        recording_name = clip_name
            else:
                clip["end"] = max(clip["end"], now + CLIP_POST_SECONDS)
            clip_name = clip["name"] if clip else None

            for kind, tid, msg in fired:
                add_event(kind, tid, msg, snap_name, clip_name)
                send_telegram("SentinelEye alert: " + msg)
                print(f"[{datetime.now():%H:%M:%S}] {kind.upper()}: {msg}")
            cleanup_evidence()

        # Write the current frame to the clip; finish it when time is up
        if clip is not None:
            clip["writer"].write(view)
            if now >= clip["end"]:
                clip["writer"].release()
                clip = None
                with lock:
                    recording_name = None

        # Keep the last few seconds for the "before" part of the next clip
        pre_buffer.append((now, view))
        while pre_buffer and now - pre_buffer[0][0] > CLIP_PRE_SECONDS:
            pre_buffer.popleft()

        ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()
                people_count = len(seen)
                alert_now = bool(candidates)

    if clip is not None:
        clip["writer"].release()
    cap.release()


@asynccontextmanager
async def lifespan(app):
    init_db()
    cleanup_evidence()
    threading.Thread(target=camera_loop, daemon=True).start()
    yield
    stop_event.set()


app = FastAPI(lifespan=lifespan)


# ---------------- Login ----------------
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
        data = {"people": people_count, "alert": alert_now,
                "recording": recording_name is not None, "sim": sim_after_hours}
    data["today"] = count_today()
    data["time"] = datetime.now().strftime("%H:%M:%S")
    return data


@app.get("/api/events")
def api_events(request: Request, limit: int = 40):
    if not is_logged_in(request):
        return JSONResponse({"error": "login required"}, status_code=401)
    return {"events": get_events(min(max(limit, 1), 200))}


@app.post("/api/simulate")
def simulate(request: Request):
    """Toggle pretend-after-hours mode (handy for testing in daytime)."""
    global sim_after_hours
    if not is_logged_in(request):
        return JSONResponse({"error": "login required"}, status_code=401)
    sim_after_hours = not sim_after_hours
    return {"sim": sim_after_hours}


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def evidence_file(request, folder, name, media, download):
    if not is_logged_in(request):
        return Response(status_code=401)
    if not SAFE_NAME.match(name):
        return Response(status_code=400)
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        return Response(status_code=404)
    if download:
        return FileResponse(path, media_type=media, filename=name)
    return FileResponse(path, media_type=media)


@app.get("/evidence/snapshots/{name}")
def get_snapshot_file(name: str, request: Request, dl: int = 0):
    return evidence_file(request, SNAP_DIR, name, "image/jpeg", dl == 1)


@app.get("/evidence/clips/{name}")
def get_clip_file(name: str, request: Request, dl: int = 0):
    return evidence_file(request, CLIP_DIR, name, "video/mp4", dl == 1)


# ---------------- Web pages ----------------
STYLE = """
<style>
:root{--bg:#f4f6f8;--card:#ffffff;--text:#1a202c;--muted:#64748b;--accent:#0fa3b1;--border:#d8dee6;--danger:#d64545}
[data-theme=dark]{--bg:#0b1220;--card:#131c2e;--text:#e6edf6;--muted:#93a4bd;--border:#243049}
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;
  background:var(--card);border-bottom:1px solid var(--border);flex-wrap:wrap;gap:8px}
h1{font-size:18px;margin:0;color:var(--accent)}
h2{font-size:16px;margin:0 0 10px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;
  font-size:14px;cursor:pointer;margin:2px}
button.ghost{background:transparent;color:var(--text);border:1px solid var(--border)}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.wrap{max-width:900px;margin:18px auto;padding:0 14px;display:grid;gap:14px}
img.video{width:100%;border-radius:10px;background:#000;min-height:200px;display:block}
.alertbar{display:none;background:var(--danger);color:#fff;font-weight:600;padding:8px 12px;
  border-radius:8px;margin-bottom:10px}
.alertbar.on{display:block}
.row{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
.stat{flex:1;min-width:120px;text-align:center;padding:10px}
.stat b{display:block;font-size:24px;color:var(--accent)}
.stat span{color:var(--muted);font-size:13px}
input{width:100%;padding:11px;border-radius:8px;border:1px solid var(--border);
  background:var(--bg);color:var(--text);font-size:16px;margin:10px 0}
.err{color:var(--danger);min-height:20px;font-size:14px}
.login{max-width:360px;margin:12vh auto;text-align:center}
.muted{color:var(--muted)}
.ev{display:flex;gap:12px;padding:10px 0;border-top:1px solid var(--border)}
.ev img{width:120px;height:90px;object-fit:cover;border-radius:8px;background:#000}
.info{flex:1;min-width:0;font-size:14px}
.info small{color:var(--muted);display:block;margin-top:2px}
.info a{color:var(--accent);margin-right:10px;font-size:13px}
.tag{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;
  margin-bottom:4px;color:#fff;background:#64748b}
.tag.zone{background:#d64545}.tag.fall{background:#b3261e}
.tag.loiter{background:#d97706}.tag.after{background:#6d4fc2}
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
  <p class="muted">Enter the access key shown on the camera device.</p>
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
    <div id="alertbar" class="alertbar">ALERT: unusual activity detected</div>
    <img id="vid" class="video" src="/video" alt="Live video (starting...)">
    <div class="row">
      <div class="card stat"><b id="people">0</b><span>People in view</span></div>
      <div class="card stat"><b id="today">0</b><span>Alerts today</span></div>
      <div class="card stat"><b id="clock">--:--</b><span>Server time</span></div>
      <div class="card stat"><b id="rec">-</b><span>Recording</span></div>
    </div>
    <div class="row">
      <button onclick="location.href='/snapshot'">Snapshot</button>
      <button onclick="document.getElementById('vid').requestFullscreen()">Fullscreen</button>
      <button class="ghost" id="simbtn" onclick="toggleSim()">Simulate after-hours: OFF</button>
    </div>
  </div>
  <div class="card">
    <h2>Event history</h2>
    <div id="events"><p class="muted">Loading...</p></div>
  </div>
</div>
<script>
const KIND = {zone:'Restricted zone', loiter:'Loitering', fall:'Fall', after:'After-hours'};
let lastSig = '';

async function poll(){
  try{
    const r = await fetch('/api/status');
    if(r.status===401){ location.reload(); return; }
    const d = await r.json();
    document.getElementById('people').textContent = d.people;
    document.getElementById('today').textContent = d.today;
    document.getElementById('clock').textContent = d.time;
    document.getElementById('rec').textContent = d.recording ? 'REC' : '-';
    document.getElementById('alertbar').classList.toggle('on', d.alert);
    document.getElementById('simbtn').textContent = 'Simulate after-hours: ' + (d.sim ? 'ON' : 'OFF');
  }catch(e){}
}

async function loadEvents(){
  try{
    const r = await fetch('/api/events?limit=40');
    if(r.status===401){ location.reload(); return; }
    const d = await r.json();
    const sig = JSON.stringify(d.events.map(e => [e.id, !!e.clip, e.recording]));
    if(sig === lastSig) return;
    lastSig = sig;
    const box = document.getElementById('events');
    box.textContent = '';
    if(!d.events.length){
      const p = document.createElement('p'); p.className = 'muted';
      p.textContent = 'No events yet.'; box.appendChild(p); return;
    }
    for(const e of d.events){
      const item = document.createElement('div'); item.className = 'ev';
      if(e.snapshot){
        const a = document.createElement('a');
        a.href = '/evidence/snapshots/' + encodeURIComponent(e.snapshot); a.target = '_blank';
        const im = document.createElement('img');
        im.src = a.href; im.loading = 'lazy'; im.alt = 'snapshot';
        a.appendChild(im); item.appendChild(a);
      }
      const info = document.createElement('div'); info.className = 'info';
      const tag = document.createElement('span');
      tag.className = 'tag ' + e.kind; tag.textContent = KIND[e.kind] || e.kind;
      const msg = document.createElement('div'); msg.textContent = e.message;
      const when = document.createElement('small'); when.textContent = e.ts;
      info.append(tag, msg, when);
      if(e.clip){
        const base = '/evidence/clips/' + encodeURIComponent(e.clip);
        if(e.recording){
          const s = document.createElement('small'); s.textContent = 'Clip is still recording...';
          info.appendChild(s);
        } else {
          const play = document.createElement('a'); play.href = base; play.target = '_blank'; play.textContent = 'Open clip';
          const dl = document.createElement('a'); dl.href = base + '?dl=1'; dl.textContent = 'Download clip';
          info.append(play, dl);
        }
      }
      item.appendChild(info); box.appendChild(item);
    }
  }catch(e){}
}

async function toggleSim(){ await fetch('/api/simulate',{method:'POST'}); poll(); }
async function logout(){ await fetch('/api/logout',{method:'POST'}); location.reload(); }
setInterval(poll, 1000); setInterval(loadEvents, 2000); poll(); loadEvents();
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
    print("=" * 52)
    print(" SentinelEye server starting")
    print(f" On this laptop : http://localhost:{PORT}")
    print(f" On other devices (same Wi-Fi): http://{ip}:{PORT}")
    print(f" ACCESS KEY     : {ACCESS_KEY}")
    print(f" Telegram alerts: {'ON' if TELEGRAM else 'off (optional)'}")
    print("=" * 52)
    uvicorn.run(app, host="0.0.0.0", port=PORT)