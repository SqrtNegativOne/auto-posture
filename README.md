# Auto Posture Monitor

Runs silently in your Windows system tray and sends a toast notification whenever you've been slouching for too long. If you get up, it pauses automatically.

## How it works

1. On first launch the app asks you to **sit up straight for a few seconds** while it captures your personal baseline.
2. It then monitors your posture at 2 fps (low CPU) using your webcam.
3. If you slouch for more than **8 seconds** you get a Windows notification.
4. Notifications cool down to a maximum of **once every 5 minutes** so you're not spammed.
5. If no pose is detected (you walked away), monitoring pauses until you return.

### Tray icon colours

| Colour | Meaning |
|--------|---------|
| Gray   | Starting / user away |
| Yellow | Calibrating — sit up straight! |
| Green  | Good posture |
| Orange | Slouching (timer running) |
| Red    | Alert just sent |

Right-click the tray icon to **recalibrate**, **open a debug window**, or **quit**.

---

## Requirements

- Windows 10 / 11
- Python 3.10 or newer
- A webcam visible from the front (laptop built-in webcam works great)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

---

## Installation

### 1 — Install uv (if you don't have it)

```powershell
winget install astral-sh.uv
```

Or with pip:

```powershell
pip install uv
```

### 2 — Clone / download the project

```powershell
git clone <repo-url>
cd auto-posture
```

### 3 — Create a virtual environment and install dependencies

```powershell
uv venv
uv pip install -e .
```

This creates a `.venv` folder and installs all dependencies listed in `pyproject.toml`.

---

## Running the app

### Activate the venv and run

```powershell
.venv\Scripts\activate
python posture_monitor.py
```

The app minimises to the system tray immediately. There is no console window to leave open.

### Run without activating the venv

```powershell
uv run python posture_monitor.py
```

### Run at Windows startup (optional)

1. Press `Win + R`, type `shell:startup`, press Enter.
2. Create a shortcut to the following command in that folder:

```
"C:\path\to\auto-posture\.venv\Scripts\pythonw.exe" "C:\path\to\auto-posture\posture_monitor.py"
```

Using `pythonw.exe` hides the console window completely.

---

## Configuration

Settings are stored in `~/.posture_monitor.json` and are created automatically on first run.

| Key | Default | Description |
|-----|---------|-------------|
| `camera_index` | `0` | Which webcam to use (0 = default) |
| `slouch_hold_seconds` | `8` | Seconds of continuous slouching before an alert |
| `notification_cooldown_seconds` | `300` | Minimum gap between notifications |
| `neck_drop_threshold` | `0.18` | How much the head must drop (normalised) to count as slouching — lower = more sensitive |
| `ear_forward_threshold` | `0.28` | How much the ears must shift forward from baseline |
| `calibration_samples` | `25` | Good-posture frames to average for the baseline |
| `process_fps` | `2.0` | Frames analysed per second |
| `away_timeout_seconds` | `5.0` | Seconds without a detected pose before pausing |
| `show_debug_window` | `false` | Toggle from the tray menu — shows metrics live |

---

## Logs

Logs are written to `~/.posture_monitor.log` (rotated at 1 MB, last 3 kept).
View the live log:

```powershell
Get-Content $HOME\.posture_monitor.log -Wait
```

---

## Troubleshooting

**No camera found**
Set `"camera_index"` in the config file to a different value (try 1, 2, …).

**Too many false alerts / not sensitive enough**
Adjust `neck_drop_threshold` — lower values = more sensitive. After changing, recalibrate via the tray menu.

**MediaPipe not available**
The app falls back to a simpler face-detection algorithm. Re-install with:
```powershell
uv pip install mediapipe
```

**Notifications don't appear**
Ensure Windows Focus Assist / Do Not Disturb is not blocking notifications from Python apps.
