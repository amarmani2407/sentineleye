# SentinelEye

**AI-powered anomaly and suspicious-activity detection for local-area monitoring. 100% software, no extra hardware, no cloud.**

SentinelEye turns the built-in camera of a laptop into a smart monitor. A Python engine detects and tracks people, applies activity rules, saves evidence when something unusual happens, and serves a live dashboard that any phone or laptop on the same Wi-Fi can open in a browser after entering an access key.

<!-- Add screenshots here, for example: ![Dashboard](docs/dashboard.png) -->

## Features

| Feature | Status |
|---|---|
| Live person detection and tracking with persistent IDs (YOLOv8 + ByteTrack) | Working |
| Restricted-zone intrusion alert | Working |
| Loitering alert (staying in one spot too long) | Working |
| After-hours presence alert (default 10 PM to 6 AM) | Working |
| Fall detection (pose-based torso angle) | Working, rule-based |
| Web dashboard with live video, people count and event history | Working |
| Access-key login with session tokens, rate limiting and lockout | Working |
| Dark and light theme | Working |
| Snapshot and short video clip saved for each alert | Working |
| SQLite event log and automatic deletion of the oldest evidence | Working |
| Optional Telegram alerts | Working (needs internet) |
| Learned scene-anomaly detector (autoencoder trained on your own normal footage) | Optional, experimental |

## How it works

```
Camera -> YOLOv8-pose (people + body points) -> ByteTrack (IDs, trails)
       -> Rules: zone, loiter, after-hours, fall
       -> Optional autoencoder: "is this scene unusual?"
       -> Alerts: snapshot + clip + SQLite log (+ Telegram)
       -> FastAPI server -> browser dashboard (MJPEG stream, access key)
```

## Install (Windows)

1. Install Python 3.11 or newer and Git.
2. Open a terminal in the project folder and run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

## Run

```
python server.py
```

The terminal prints a local address, a network address and your **access key**.

- On the laptop: open `http://localhost:8000`
- On a phone or another laptop on the same Wi-Fi: open the network address shown (like `http://192.168.x.x:8000`)
- Enter the access key to see the live view.

Some college or hostel Wi-Fi networks stop devices from reaching each other. If your phone cannot connect, use the phone's hotspot for both devices.

## Optional: learned anomaly detector

```
python collect_normal.py      # records ~10 minutes of NORMAL scene (tiny 64x48 grey pictures)
python train_anomaly.py       # trains a small autoencoder on your laptop
python test_anomaly.py        # live score check
python server.py              # now also raises "Unusual scene" alerts
```

Keep the camera fixed. Retrain if you move it or the lighting changes a lot.

## Settings

Edit the block at the top of `server.py`:

| Setting | Meaning |
|---|---|
| `ZONE_FRACTIONS` | Corners of the restricted zone, as fractions of the screen |
| `LOITER_SECONDS` | How long staying still counts as loitering |
| `FALL_ANGLE`, `FALL_HOLD_SECONDS` | Fall sensitivity |
| `AFTER_HOURS_START`, `AFTER_HOURS_END` | Quiet hours |
| `COOLDOWNS` | Seconds before the same alert repeats |
| `MAX_EVIDENCE_MB` | Storage limit; oldest files are deleted first |
| `ANOMALY_THRESHOLD_SCALE` | Sensitivity of the learned detector |

Telegram is optional: put the bot token on line 1 and the chat ID on line 2 of `telegram.txt`.

## Project structure

```
server.py            Main app: camera, AI, rules, alerts, web server
anomaly_model.py     Autoencoder model and scoring (optional feature)
collect_normal.py    Record normal footage for training
train_anomaly.py     Train the autoencoder
test_anomaly.py      Live anomaly-score test
examples/            Step-by-step learning scripts (detection, tracking, rules, fall)
requirements.txt     Python dependencies
```

## Privacy and security

- All processing happens on your own machine. No video is sent to any cloud service.
- Evidence (`evidence/`, `events.db`) stays local and is excluded from Git.
- Viewers must enter an access key. Wrong attempts are rate-limited, and sessions expire after 12 hours.
- Telegram, if enabled, sends only a short text message.

## Known limitations

- Traffic between the laptop and viewers uses plain HTTP, so use it only on a network you trust. HTTPS is planned.
- The access key is stored in plain text in `access_key.txt` (kept out of Git).
- Fall detection is a simple rule. Lying on a sofa looks the same as a fall.
- Video clips may need to be downloaded and opened in a player such as VLC, depending on the codec available on the machine.
- The learned detector flags "different from normal", not "dangerous", and is sensitive to camera movement and lighting.
- Streaming uses MJPEG, which has higher latency than WebRTC.
- Accuracy drops in dim light and with partial views of a person.

## Roadmap

- HTTPS with a self-signed certificate and a hashed access key
- Phone as the camera (browser camera page), then multi-camera support
- WebRTC streaming and two-way audio
- Telegram photos, detection zones drawn on the dashboard, and better trained models

## Tech stack

Python, OpenCV, Ultralytics YOLOv8 (pose) with ByteTrack, PyTorch, FastAPI, Uvicorn, SQLite, HTML/CSS/JavaScript.
