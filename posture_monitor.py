#!/usr/bin/env python3
"""
Auto Posture Monitor
====================
Runs silently in the Windows system tray and sends a notification whenever
you've been slouching for too long. Getting up is fine — if no pose is
detected the monitor pauses automatically.

Tray icon colours
-----------------
  Gray   → starting / user away
  Yellow → calibrating (sit up straight!)
  Green  → good posture
  Orange → slouching (timer running)
  Red    → alert just sent
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import urllib.request

import cv2
import pystray
from loguru import logger
from PIL import Image, ImageDraw
from winotify import Notification

# ── Optional MediaPipe (Tasks API, 0.10+) ─────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
APP_NAME = "Posture Monitor"
_HOME = Path.home()
CONFIG_PATH = _HOME / ".posture_monitor.json"
LOG_PATH    = _HOME / ".posture_monitor.log"
MODEL_PATH  = _HOME / ".posture_monitor_model.task"

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# ── Default configuration ──────────────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    # Camera
    "camera_index": 0,
    # How many seconds of continuous slouching before an alert fires
    "slouch_hold_seconds": 8,
    # Minimum seconds between consecutive notifications (5 min default)
    "notification_cooldown_seconds": 300,
    # --- MediaPipe thresholds (normalised by shoulder width) ---
    # How much the nose must drop below the calibrated height to count as a slouch
    "neck_drop_threshold": 0.18,
    # How much the ears must shift horizontally from the calibrated offset
    "ear_forward_threshold": 0.28,
    # --- Face-cascade fallback threshold ---
    # Normalised face-centre-y drop that counts as slouching
    "face_drop_threshold": 0.10,
    # How many good-posture frames to average for the baseline
    "calibration_samples": 25,
    # Frames to analyse per second (keeps CPU low)
    "process_fps": 2.0,
    # How long (seconds) without a detected pose before we consider the user away
    "away_timeout_seconds": 5.0,
    # Set True from the tray to open an annotated preview window
    "show_debug_window": False,
}


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            on_disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **on_disk}
        except Exception as exc:
            logger.warning(f"Could not read config ({exc}) — using defaults")
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.debug(f"Config saved to {CONFIG_PATH}")


# ── Posture analysers ──────────────────────────────────────────────────────────

def _ensure_model() -> str:
    """Download the MediaPipe Lite pose model on first run; return its path."""
    if not MODEL_PATH.exists():
        logger.info(f"Downloading pose model → {MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        logger.success("Model downloaded")
    return str(MODEL_PATH)


class MediaPipeAnalyzer:
    """
    Uses the MediaPipe Tasks API (0.10+) with the Lite pose-landmarker model.

    Returned dict keys
    ------------------
    neck_height  : (shoulder_mid_y − nose_y) / shoulder_width   [pixels, norm]
                   Larger = head higher above shoulders = better posture.
    ear_forward  : |ear_mid_x − shoulder_mid_x| / shoulder_width
                   Grows when ears drift away from the shoulder centre line.
    """

    # Landmark indices (same as the legacy API)
    _NOSE         = 0
    _LEFT_EAR     = 7
    _RIGHT_EAR    = 8
    _LEFT_SHOULDER  = 11
    _RIGHT_SHOULDER = 12

    def __init__(self) -> None:
        model_path = _ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._start_ms   = int(time.time() * 1000)
        logger.info("MediaPipe Pose analyser ready (Tasks API)")

    def analyze(self, frame) -> Optional[dict]:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.time() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]   # first (and only) detected person
        h, w = frame.shape[:2]

        def pt(idx):
            p = lm[idx]
            return p.x * w, p.y * h, p.visibility

        l_ear = pt(self._LEFT_EAR)
        r_ear = pt(self._RIGHT_EAR)
        l_sh  = pt(self._LEFT_SHOULDER)
        r_sh  = pt(self._RIGHT_SHOULDER)
        nose  = pt(self._NOSE)

        # Require both shoulders to be confidently visible
        if min(l_sh[2], r_sh[2]) < 0.45:
            return None

        sh_mid_x = (l_sh[0] + r_sh[0]) / 2
        sh_mid_y = (l_sh[1] + r_sh[1]) / 2
        sh_width = abs(l_sh[0] - r_sh[0])

        if sh_width < 30:          # person too far away or partially in frame
            return None

        neck_height = (sh_mid_y - nose[1]) / sh_width
        ear_mid_x   = (l_ear[0] + r_ear[0]) / 2
        ear_forward = abs(ear_mid_x - sh_mid_x) / sh_width

        return {"neck_height": neck_height, "ear_forward": ear_forward}

    def close(self) -> None:
        self._landmarker.close()


class FaceCascadeAnalyzer:
    """
    Fallback when MediaPipe is unavailable.  Tracks the face-centre Y
    position (normalised 0–1) relative to a calibrated baseline.
    When you hunch, your head drops in the frame → face_centre_y rises.
    """

    def __init__(self) -> None:
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade)
        logger.info("Face-cascade fallback analyser ready")

    def analyze(self, frame) -> Optional[dict]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None

        h = frame.shape[0]
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_centre_y = (y + fh / 2) / h
        face_size     = fh / h              # proxy for distance from camera
        return {"face_centre_y": face_centre_y, "face_size": face_size}

    def close(self) -> None:
        pass


# ── Core monitor (runs in a background thread) ─────────────────────────────────

class SlouchMonitor:
    """
    Captures frames, runs the analyser, tracks slouch duration, and fires
    Windows toast notifications when the configured threshold is exceeded.
    """

    def __init__(self, config: dict, on_status_change: Callable[[str], None]) -> None:
        self.config = config
        self.on_status_change = on_status_change   # called with a colour key

        self._analyser: MediaPipeAnalyzer | FaceCascadeAnalyzer = (
            MediaPipeAnalyzer() if MEDIAPIPE_AVAILABLE else FaceCascadeAnalyzer()
        )

        # State
        self._baseline:           Optional[dict]  = None
        self._calibration_buffer: list[dict]      = []
        self._slouch_since:       Optional[float] = None
        self._last_notif_time:    float            = 0.0
        self._last_detected:      float            = 0.0
        self._status:             str              = "gray"
        self._stop_event                           = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Reset baseline and begin collecting good-posture samples."""
        self._baseline           = None
        self._calibration_buffer = []
        self._slouch_since       = None
        self._set_status("yellow")
        logger.info("Calibration started — please sit up straight")
        Notification(
            app_id=APP_NAME,
            title="Posture Monitor — Calibrating",
            msg="Sit up straight for a few seconds while we capture your baseline.",
            duration="short",
        ).show()

    def stop(self) -> None:
        self._stop_event.set()
        self._analyser.close()

    def run(self) -> None:
        """
        Main loop.  Must be called from a background thread because
        cv2.VideoCapture is blocking and we want pystray on the main thread.
        """
        cfg         = self.config
        interval    = 1.0 / cfg["process_fps"]
        away_timeout = cfg["away_timeout_seconds"]

        cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error(f"Cannot open camera index {cfg['camera_index']}")
            self._set_status("gray")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the latest frame
        logger.info("Camera opened — monitoring started")
        self.start_calibration()

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame capture failed — retrying")
                    time.sleep(0.5)
                    continue

                metrics = self._analyser.analyze(frame)

                if metrics is None:
                    away_for = time.monotonic() - self._last_detected
                    if away_for > away_timeout and self._status != "gray":
                        logger.debug(f"No pose for {away_for:.1f}s — user away")
                        self._slouch_since = None
                        self._set_status("gray")
                else:
                    self._last_detected = time.monotonic()
                    self._process_metrics(metrics)

                if cfg.get("show_debug_window") and metrics:
                    self._annotate(frame, metrics)
                    cv2.imshow("Posture Monitor — Debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, interval - elapsed))

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _process_metrics(self, metrics: dict) -> None:
        cfg = self.config

        # ── Calibration phase ────────────────────────────────────────────
        if self._baseline is None:
            self._calibration_buffer.append(metrics)
            n      = len(self._calibration_buffer)
            target = cfg["calibration_samples"]
            logger.debug(f"Calibrating {n}/{target}")

            if n >= target:
                self._baseline = {
                    k: sum(d[k] for d in self._calibration_buffer) / n
                    for k in self._calibration_buffer[0]
                }
                logger.success(f"Calibration complete — baseline: {self._baseline}")
                self._set_status("green")
                Notification(
                    app_id=APP_NAME,
                    title="Posture Monitor — Ready",
                    msg="Calibration done! I'll alert you if you slouch.",
                    duration="short",
                ).show()
            return

        # ── Slouch check ─────────────────────────────────────────────────
        is_slouching = self._compute_slouch(metrics)
        now          = time.monotonic()

        if is_slouching:
            if self._slouch_since is None:
                self._slouch_since = now
                logger.debug("Slouch timer started")
            self._set_status("orange")

            held        = now - self._slouch_since
            can_notify  = now - self._last_notif_time > cfg["notification_cooldown_seconds"]

            if held >= cfg["slouch_hold_seconds"] and can_notify:
                self._fire_notification()
                self._last_notif_time = now
                self._set_status("red")
        else:
            if self._slouch_since is not None:
                logger.debug("Good posture restored")
            self._slouch_since = None
            self._set_status("green")

    def _compute_slouch(self, metrics: dict) -> bool:
        cfg      = self.config
        baseline = self._baseline

        if "neck_height" in metrics:
            # MediaPipe path
            neck_drop   = baseline["neck_height"] - metrics["neck_height"]
            ear_drift   = metrics["ear_forward"]  - baseline["ear_forward"]
            slouching   = (
                neck_drop > cfg["neck_drop_threshold"]
                or ear_drift > cfg["ear_forward_threshold"]
            )
            logger.debug(
                f"neck_drop={neck_drop:+.3f} "
                f"ear_drift={ear_drift:+.3f} "
                f"slouching={slouching}"
            )
            return slouching

        else:
            # Face-cascade fallback path
            raw_drop  = metrics["face_centre_y"] - baseline["face_centre_y"]
            norm_drop = raw_drop / max(metrics["face_size"], 0.01)
            slouching = norm_drop > cfg["face_drop_threshold"]
            logger.debug(f"face_drop={raw_drop:+.3f} norm={norm_drop:+.3f} slouching={slouching}")
            return slouching

    def _fire_notification(self) -> None:
        Notification(
            app_id=APP_NAME,
            title="Sit up straight!",
            msg="You've been slouching — roll your shoulders back and lift your chin.",
            duration="long",
        ).show()
        logger.info("Slouch alert notification sent")

    def _annotate(self, frame, metrics: dict) -> None:
        status_text = f"Status: {self._status}"
        metrics_text = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        cv2.putText(frame, metrics_text,  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        cv2.putText(frame, status_text,   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        if self._baseline:
            base_text = "  ".join(f"base_{k}={v:.3f}" for k, v in self._baseline.items())
            cv2.putText(frame, base_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _set_status(self, status: str) -> None:
        if status != self._status:
            self._status = status
            logger.debug(f"Status → {status}")
            self.on_status_change(status)


# ── System tray ────────────────────────────────────────────────────────────────

_ICON_COLOURS = {
    "gray":   (130, 130, 130),
    "yellow": (255, 200,   0),
    "green":  ( 50, 200,  80),
    "orange": (255, 140,   0),
    "red":    (220,  50,  50),
}

_STATUS_LABELS = {
    "gray":   "Away / Starting",
    "yellow": "Calibrating…",
    "green":  "Good posture",
    "orange": "Slouching…",
    "red":    "Alert sent",
}


def _make_icon(status: str) -> Image.Image:
    colour = _ICON_COLOURS.get(status, _ICON_COLOURS["gray"])
    img    = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(img)
    draw.ellipse((4, 4, 60, 60), fill=colour)
    return img


def build_tray(monitor: SlouchMonitor, config: dict) -> pystray.Icon:
    """
    Build the pystray Icon.  Wire the monitor's status-change callback so
    the tray icon colour updates live.  Must run on the main thread.
    """

    # Forward-reference trick: tray is assigned below, closure captures it by name
    tray_holder: list[pystray.Icon] = []  # mutable cell so the callback can reference it

    def update_icon(status: str) -> None:
        if tray_holder:
            tray_holder[0].icon  = _make_icon(status)
            tray_holder[0].title = f"{APP_NAME} — {_STATUS_LABELS.get(status, status)}"

    monitor.on_status_change = update_icon

    def on_calibrate(_icon, _item) -> None:
        monitor.start_calibration()

    def on_toggle_debug(_icon, _item) -> None:
        config["show_debug_window"] = not config.get("show_debug_window", False)
        save_config(config)
        state = "on" if config["show_debug_window"] else "off"
        logger.info(f"Debug window turned {state}")

    def on_quit(_icon, _item) -> None:
        monitor.stop()
        _icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Calibrate (sit up straight now)", on_calibrate),
        pystray.MenuItem("Toggle debug window",              on_toggle_debug),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit",                             on_quit),
    )

    tray = pystray.Icon(
        name="posture_monitor",
        icon=_make_icon("gray"),
        title=f"{APP_NAME} — Starting…",
        menu=menu,
    )
    tray_holder.append(tray)
    return tray


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Logging ───────────────────────────────────────────────────────────
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        LOG_PATH,
        level="DEBUG",
        rotation="1 MB",
        retention=3,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    logger.info(f"{APP_NAME} starting — logs → {LOG_PATH}")

    config  = load_config()
    monitor = SlouchMonitor(config, on_status_change=lambda _: None)

    thread = threading.Thread(target=monitor.run, daemon=True, name="PostureMonitor")
    thread.start()

    tray = build_tray(monitor, config)
    logger.info("System tray active — right-click the tray icon to control the app")

    try:
        tray.run()   # blocks on the main thread until Quit is chosen
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        monitor.stop()


if __name__ == "__main__":
    main()
