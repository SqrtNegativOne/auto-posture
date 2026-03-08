# Auto Posture Monitor — CLAUDE.md

## Environment
- Python 3.13 | Windows 11
- Package manager: `uv` — use `uv venv` + `uv pip install -e .`
- Run: `.venv/Scripts/python.exe posture_monitor.py` or `uv run python posture_monitor.py`

## Conventions
- Logging: loguru — `logger.add(LOG_PATH, rotation="1 MB", retention=3)`
- Notifications: winotify (`from winotify import Notification`)
- Single-file app: all logic lives in `posture_monitor.py`

## MediaPipe gotcha (critical)
- Version 0.10+ removed `mp.solutions` entirely — do NOT use it
- Correct API: `from mediapipe.tasks.python import vision as mp_vision`
- Requires a `.task` model file downloaded at runtime (see `_ensure_model()`)
- Use `RunningMode.VIDEO` for live webcam — `IMAGE` mode lacks temporal tracking

## Key file locations (runtime, not in repo)
- Config:  `~/.posture_monitor.json`
- Log:     `~/.posture_monitor.log`
- Model:   `~/.posture_monitor_model.task` (auto-downloaded on first run)

## Threading
- `pystray` must run on the main thread — detection runs in a daemon thread
- `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` prevents stale webcam frames
