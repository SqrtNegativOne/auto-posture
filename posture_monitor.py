#!/usr/bin/env python3
"""
Auto Posture Monitor
====================
Runs silently in the Windows system tray and sends a notification whenever
you've been slouching for too long. Getting up is fine — if no pose is
detected the monitor pauses automatically.

Left-click the tray icon to open a live preview window showing your webcam
feed and posture status.  Close the preview to tuck it back into the tray.

Tray icon colours
-----------------
  Gray   -> starting / user away
  Yellow -> calibrating (sit up straight!)
  Green  -> good posture
  Orange -> slouching (timer running)
  Red    -> alert just sent
"""

from __future__ import annotations

import json
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Callable, Optional

import urllib.request

import cv2
import pystray
from loguru import logger
from PIL import Image, ImageDraw, ImageTk
from winotify import Notification

# -- Optional MediaPipe (Tasks API, 0.10+) ------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# -- Paths ---------------------------------------------------------------------
APP_NAME = "Posture Monitor"
_HOME = Path.home()
CONFIG_PATH = _HOME / ".posture_monitor.json"
LOG_PATH    = _HOME / ".posture_monitor.log"
MODEL_PATH  = _HOME / ".posture_monitor_model.task"

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# -- Default configuration ----------------------------------------------------
DEFAULT_CONFIG: dict = {
    # Camera
    "camera_index": 0,
    # How many seconds of continuous slouching before an alert fires
    "slouch_hold_seconds": 8,
    # Minimum seconds between consecutive notifications (5 min default)
    "notification_cooldown_seconds": 300,
    # --- MediaPipe thresholds (normalised by shoulder width) ---
    "neck_drop_threshold": 0.18,
    "ear_forward_threshold": 0.28,
    # --- Face-cascade fallback threshold ---
    "face_drop_threshold": 0.10,
    # How many good-posture frames to average for the baseline
    "calibration_samples": 25,
    # Frames to analyse per second (keeps CPU low)
    "process_fps": 2.0,
    # How long (seconds) without a detected pose before we consider the user away
    "away_timeout_seconds": 5.0,
}


# -- Config helpers ------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            on_disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **on_disk}
        except Exception as exc:
            logger.warning(f"Could not read config ({exc}) -- using defaults")
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.debug(f"Config saved to {CONFIG_PATH}")


# -- Posture analysers ---------------------------------------------------------

def _ensure_model() -> str:
    """Download the MediaPipe Lite pose model on first run; return its path."""
    if not MODEL_PATH.exists():
        logger.info(f"Downloading pose model -> {MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        logger.success("Model downloaded")
    return str(MODEL_PATH)


class MediaPipeAnalyzer:
    """
    Uses the MediaPipe Tasks API (0.10+) with the Lite pose-landmarker model.

    Returned dict keys
    ------------------
    neck_height  : (shoulder_mid_y - nose_y) / shoulder_width   [pixels, norm]
    ear_forward  : |ear_mid_x - shoulder_mid_x| / shoulder_width
    """

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

        lm = result.pose_landmarks[0]
        h, w = frame.shape[:2]

        def pt(idx):
            p = lm[idx]
            return p.x * w, p.y * h, p.visibility

        l_ear = pt(self._LEFT_EAR)
        r_ear = pt(self._RIGHT_EAR)
        l_sh  = pt(self._LEFT_SHOULDER)
        r_sh  = pt(self._RIGHT_SHOULDER)
        nose  = pt(self._NOSE)

        if min(l_sh[2], r_sh[2]) < 0.45:
            return None

        sh_mid_x = (l_sh[0] + r_sh[0]) / 2
        sh_mid_y = (l_sh[1] + r_sh[1]) / 2
        sh_width = abs(l_sh[0] - r_sh[0])

        if sh_width < 30:
            return None

        neck_height = (sh_mid_y - nose[1]) / sh_width
        ear_mid_x   = (l_ear[0] + r_ear[0]) / 2
        ear_forward = abs(ear_mid_x - sh_mid_x) / sh_width

        return {"neck_height": neck_height, "ear_forward": ear_forward}

    def close(self) -> None:
        self._landmarker.close()


class FaceCascadeAnalyzer:
    """Fallback when MediaPipe is unavailable."""

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
        face_size     = fh / h
        return {"face_centre_y": face_centre_y, "face_size": face_size}

    def close(self) -> None:
        pass


# -- Core monitor (runs in a background thread) -------------------------------

class SlouchMonitor:
    """
    Captures frames, runs the analyser, tracks slouch duration, and fires
    Windows toast notifications when the configured threshold is exceeded.
    """

    def __init__(self, config: dict, on_status_change: Callable[[str], None]) -> None:
        self.config = config
        self.on_status_change = on_status_change

        self._analyser: MediaPipeAnalyzer | FaceCascadeAnalyzer = (
            MediaPipeAnalyzer() if MEDIAPIPE_AVAILABLE else FaceCascadeAnalyzer()
        )

        # Posture state
        self._baseline:           Optional[dict]  = None
        self._calibration_buffer: list[dict]      = []
        self._slouch_since:       Optional[float] = None
        self._last_notif_time:    float            = 0.0
        self._last_detected:      float            = 0.0
        self._status:             str              = "gray"
        self._stop_event                           = threading.Event()

        # Frame sharing for the preview window (written by monitor thread,
        # read by tkinter thread — protected by _frame_lock)
        self._frame_lock:     threading.Lock      = threading.Lock()
        self._latest_frame                         = None   # annotated BGR frame
        self._latest_metrics: Optional[dict]       = None
        self._preview_open:   bool                 = False  # set by PreviewWindow

    # -- Public API ------------------------------------------------------------

    def get_latest_frame(self) -> tuple:
        """Return (frame, metrics, status) for the preview window."""
        with self._frame_lock:
            return self._latest_frame, self._latest_metrics, self._status

    def start_calibration(self) -> None:
        """Reset baseline and begin collecting good-posture samples."""
        self._baseline           = None
        self._calibration_buffer = []
        self._slouch_since       = None
        self._set_status("yellow")
        logger.info("Calibration started -- please sit up straight")
        Notification(
            app_id=APP_NAME,
            title="Posture Monitor -- Calibrating",
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
        cfg              = self.config
        analysis_interval = 1.0 / cfg["process_fps"]
        away_timeout     = cfg["away_timeout_seconds"]

        cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error(f"Cannot open camera index {cfg['camera_index']}")
            self._set_status("gray")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Camera opened -- monitoring started")
        self.start_calibration()

        last_analysis:  float         = 0.0
        cached_metrics: Optional[dict] = None

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame capture failed -- retrying")
                    time.sleep(0.5)
                    continue

                # --- Run pose analysis at the configured (low) rate -----------
                if t0 - last_analysis >= analysis_interval:
                    last_analysis = t0
                    metrics = self._analyser.analyze(frame)
                    cached_metrics = metrics

                    if metrics is None:
                        if t0 - self._last_detected > away_timeout and self._status != "gray":
                            logger.debug("No pose detected -- user away")
                            self._slouch_since = None
                            self._set_status("gray")
                    else:
                        self._last_detected = t0
                        self._process_metrics(metrics)

                # --- Store annotated frame for the preview window -------------
                display = frame.copy()
                self._annotate(display, cached_metrics)
                with self._frame_lock:
                    self._latest_frame   = display
                    self._latest_metrics = cached_metrics

                # When the preview window is open, capture at ~18 FPS for
                # smooth video.  Otherwise stay at the analysis rate to
                # keep CPU usage low.
                target_fps = 18.0 if self._preview_open else cfg["process_fps"]
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, (1.0 / target_fps) - elapsed))

        finally:
            cap.release()
            logger.info("Camera released")

    # -- Internal helpers ------------------------------------------------------

    def _process_metrics(self, metrics: dict) -> None:
        cfg = self.config

        # Calibration phase
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
                logger.success(f"Calibration complete -- baseline: {self._baseline}")
                self._set_status("green")
                Notification(
                    app_id=APP_NAME,
                    title="Posture Monitor -- Ready",
                    msg="Calibration done! I'll alert you if you slouch.",
                    duration="short",
                ).show()
            return

        # Slouch check
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
            raw_drop  = metrics["face_centre_y"] - baseline["face_centre_y"]
            norm_drop = raw_drop / max(metrics["face_size"], 0.01)
            slouching = norm_drop > cfg["face_drop_threshold"]
            logger.debug(f"face_drop={raw_drop:+.3f} norm={norm_drop:+.3f} slouching={slouching}")
            return slouching

    def _fire_notification(self) -> None:
        Notification(
            app_id=APP_NAME,
            title="Sit up straight!",
            msg="You've been slouching -- roll your shoulders back and lift your chin.",
            duration="long",
        ).show()
        logger.info("Slouch alert notification sent")

    def _annotate(self, frame, metrics: Optional[dict]) -> None:
        """Draw a status overlay on *frame* (mutates in-place)."""
        h, w = frame.shape[:2]
        status = self._status

        # BGR colours for OpenCV drawing
        bgr_colours = {
            "gray":   (130, 130, 130),
            "yellow": (0, 200, 255),
            "green":  (80, 200, 50),
            "orange": (0, 140, 255),
            "red":    (50, 50, 220),
        }
        colour = bgr_colours.get(status, (130, 130, 130))

        # Coloured border around the whole frame
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, 3)

        # Semi-transparent dark bar along the bottom
        bar_h = 58
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Status dot + label
        label = _STATUS_LABELS.get(status, status)
        cv2.circle(frame, (20, h - bar_h + 20), 7, colour, -1)
        cv2.putText(frame, label, (36, h - bar_h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Slouch timer countdown (visible while slouching)
        if self._slouch_since is not None:
            held = time.monotonic() - self._slouch_since
            threshold = self.config["slouch_hold_seconds"]
            timer_text = f"{held:.0f}s / {threshold}s"
            cv2.putText(frame, timer_text, (w - 150, h - bar_h + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Raw metric values
        if metrics:
            metrics_text = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            cv2.putText(frame, metrics_text, (14, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    def _set_status(self, status: str) -> None:
        if status != self._status:
            self._status = status
            logger.debug(f"Status -> {status}")
            self.on_status_change(status)


# -- Preview window (tkinter, runs in its own daemon thread) -------------------

class PreviewWindow:
    """
    A live-view window that shows the webcam feed with posture overlay.
    Opens in its own thread so it doesn't block the pystray main loop.
    Closing the window simply hides it back to the tray.
    """

    def __init__(self, monitor: SlouchMonitor) -> None:
        self._monitor = monitor
        self._root: Optional[tk.Tk] = None
        self._thread: Optional[threading.Thread] = None
        self._photo = None          # prevent garbage collection of PhotoImage

    def toggle(self) -> None:
        """Open the preview, or bring an existing one to the front."""
        if self._root is not None:
            try:
                self._root.after(0, self._root.lift)
                return
            except (tk.TclError, RuntimeError):
                pass                # window was already destroyed
        # Don't spawn a second thread if one is still starting up
        if self._thread is not None and self._thread.is_alive():
            return
        self._monitor._preview_open = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PreviewWindow"
        )
        self._thread.start()

    def destroy(self) -> None:
        """Tear down the preview (called on app quit)."""
        self._monitor._preview_open = False
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except (tk.TclError, RuntimeError):
                pass

    # -- Private ---------------------------------------------------------------

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("Posture Monitor")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Window icon (reuse the tray-icon helper)
        try:
            icon_img = _make_icon("green")
            root.iconphoto(False, ImageTk.PhotoImage(icon_img))
        except Exception:
            pass

        # -- Webcam canvas -----------------------------------------------------
        self._canvas = tk.Canvas(
            root, width=640, height=480, bg="#000000", highlightthickness=0,
        )
        self._canvas.pack(padx=10, pady=(10, 4))

        # -- Status bar --------------------------------------------------------
        bar = tk.Frame(root, bg="#1a1a2e")
        bar.pack(fill="x", padx=14, pady=(4, 10))

        self._status_lbl = tk.Label(
            bar, text="  Starting...",
            font=("Segoe UI", 13, "bold"), fg="#888888", bg="#1a1a2e", anchor="w",
        )
        self._status_lbl.pack(fill="x")

        self._hint_lbl = tk.Label(
            bar, text="Close this window to minimize back to tray.",
            font=("Segoe UI", 8), fg="#555555", bg="#1a1a2e", anchor="w",
        )
        self._hint_lbl.pack(fill="x")

        # Start polling for new frames
        self._poll()
        root.mainloop()

        # Cleanup after mainloop exits
        self._root = None
        self._monitor._preview_open = False

    def _poll(self) -> None:
        """Fetch the latest frame from the monitor and refresh the canvas."""
        if self._root is None:
            return
        try:
            frame, _metrics, status = self._monitor.get_latest_frame()

            if frame is not None:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((640, 480), Image.LANCZOS)
                self._photo = ImageTk.PhotoImage(pil_img)
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

            # Colour-code the status label
            fg_map = {
                "gray": "#888888", "yellow": "#ffc800", "green": "#32c850",
                "orange": "#ff8c00", "red": "#dc3232",
            }
            label_text = _STATUS_LABELS.get(status, "Unknown")
            self._status_lbl.configure(
                text=f"  {label_text}", fg=fg_map.get(status, "#888888"),
            )

            self._root.after(55, self._poll)     # ~18 FPS display refresh
        except tk.TclError:
            # Window was destroyed between iterations
            self._root = None

    def _on_close(self) -> None:
        self._monitor._preview_open = False
        if self._root is not None:
            self._root.destroy()
            self._root = None


# -- System tray ---------------------------------------------------------------

_ICON_COLOURS = {
    "gray":   (130, 130, 130),
    "yellow": (255, 200,   0),
    "green":  ( 50, 200,  80),
    "orange": (255, 140,   0),
    "red":    (220,  50,  50),
}

_STATUS_LABELS = {
    "gray":   "Away / Starting",
    "yellow": "Calibrating...",
    "green":  "Good posture",
    "orange": "Slouching...",
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
    Build the pystray Icon with a left-click action that opens the preview
    window and a right-click context menu for calibration / quit.
    """
    tray_holder: list[pystray.Icon] = []
    preview = PreviewWindow(monitor)

    def update_icon(status: str) -> None:
        if tray_holder:
            tray_holder[0].icon  = _make_icon(status)
            tray_holder[0].title = f"{APP_NAME} -- {_STATUS_LABELS.get(status, status)}"

    monitor.on_status_change = update_icon

    def on_open(_icon, _item) -> None:
        preview.toggle()

    def on_calibrate(_icon, _item) -> None:
        monitor.start_calibration()

    def on_quit(_icon, _item) -> None:
        preview.destroy()
        monitor.stop()
        _icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open Monitor",                    on_open,     default=True),
        pystray.MenuItem("Calibrate (sit up straight now)", on_calibrate),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit",                            on_quit),
    )

    tray = pystray.Icon(
        name="posture_monitor",
        icon=_make_icon("gray"),
        title=f"{APP_NAME} -- Starting...",
        menu=menu,
    )
    tray_holder.append(tray)
    return tray


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    # -- Logging ---------------------------------------------------------------
    logger.remove()

    # sys.stderr is None when running via pythonw.exe (no console).
    if sys.stderr is not None:
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
    logger.info(f"{APP_NAME} starting -- logs -> {LOG_PATH}")

    config  = load_config()
    monitor = SlouchMonitor(config, on_status_change=lambda _: None)

    thread = threading.Thread(target=monitor.run, daemon=True, name="PostureMonitor")
    thread.start()

    tray = build_tray(monitor, config)
    logger.info("System tray active -- left-click the icon to open the monitor")

    try:
        tray.run()          # blocks on the main thread until Quit is chosen
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt -- shutting down")
        monitor.stop()


if __name__ == "__main__":
    main()
