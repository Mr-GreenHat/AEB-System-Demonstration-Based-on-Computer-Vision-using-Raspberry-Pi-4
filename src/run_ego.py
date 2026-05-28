import os
# Avoid OpenCV Qt font crash/warnings on Raspberry Pi / Python venv.
# This must be set before importing cv2.
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import time
import math
import threading
import matplotlib.pyplot as plt

from ego_sim import EgoVehicle
from ipm.webcam_distance_test import main as vision_main

# ============================================================
# RPi GPIO — imported if available; stubs used on dev machine
# ============================================================
try:
    import RPi.GPIO as GPIO
    _ON_RPI = True
except ImportError:
    _ON_RPI = False

# pigpio gives hardware-timed PWM — no servo jitter.
# RPi.GPIO software PWM is the fallback (works but servo may buzz).
# To use pigpio: sudo apt install pigpio python3-pigpio && sudo pigpiod
try:
    import pigpio as _pigpio_mod
    _pigpio_pi = _pigpio_mod.pi() if _ON_RPI else None
    _USE_PIGPIO = _ON_RPI and (_pigpio_pi is not None) and _pigpio_pi.connected
except Exception:
    _pigpio_pi  = None
    _USE_PIGPIO = False

# ============================================================
# Timing / display config
# ============================================================
DEBUG_TIMING        = True
PRINT_EVERY_N_LOOPS = 150   # ~every 3 s at 50 Hz
DISPLAY_EVERY_N     = 5     # ~10 FPS display at 50 Hz; safer for 1080p dashboard
SHOW_MATPLOTLIB_PLOTS = False  # keep False for HDMI demo; plt.show() can block closing
TV_MODE             = True   # HDMI TV mode for Sharp 2T-C42BE1 / 1080p
SCREEN_W            = 1920   # Sharp 2T-C42BE1 Full HD width
SCREEN_H            = 1080   # Sharp 2T-C42BE1 Full HD height
# 1080p dashboard layout: camera + status on top, graph + ego animation below
# Black-background presentation version with clickable EXIT button.
DASHBOARD_MODE      = True
TOP_H               = 700
BOTTOM_H            = SCREEN_H - TOP_H
CAM_W               = 1240
STATUS_W            = SCREEN_W - CAM_W
GRAPH_W             = 1240
EGO_W               = SCREEN_W - GRAPH_W
GRAPH_HISTORY_SEC   = 12.0
CAM_DISPLAY_H       = TOP_H   # kept for compatibility with old display path
USE_OPENCV_FULLSCREEN = False  # False is safer; Qt fullscreen can crash on some Raspberry Pi/OpenCV builds

# On-screen mouse/touch EXIT button. Useful when keyboard focus breaks.
EXIT_BUTTON_W       = 150
EXIT_BUTTON_H       = 64
EXIT_BUTTON_MARGIN  = 18
_exit_requested     = False


# ============================================================
# Control loop rate  — decoupled from YOLO speed
# ============================================================
CONTROL_HZ = 50
DT         = 1.0 / CONTROL_HZ   # 20 ms per control tick

# ============================================================
# World / display geometry
# ============================================================
WORLD_WIDTH  = 1000
WORLD_HEIGHT = 260
ROBOT_Y      = WORLD_HEIGHT // 2
OBJECT_Y     = WORLD_HEIGHT // 2
BOX_W        = 40
BOX_H        = 40

MIN_VALID_DISTANCE = 0.10

# ============================================================
# Init behaviour
# ============================================================
INIT_REQUIRED_SAMPLES = 8
INIT_MAX_WAIT_SEC     = 2.0

# ============================================================
# TTC / braking thresholds
# ============================================================
SAFE_TTC    = 2.6
FCW_TTC     = 1.6
PARTIAL_TTC = 0.6

FULL_BRAKE_DECEL    = 6.43
PARTIAL_BRAKE_DECEL = FULL_BRAKE_DECEL * 0.4

STOP_EPS = 0.01

# ============================================================
# Manual speed fallback (keyboard demo — no encoder needed)
# ============================================================
USE_MANUAL_SPEED  = True   # set False when encoder is wired up
MANUAL_SPEED_STEP = 0.10
MAX_DEMO_SPEED    = 5.0

manual_speed_mps = 0.0

# ============================================================
# GPIO pin assignments  (BCM numbering)
# ============================================================
PIN_MOTOR_PWM = 12   # BTS7960 PWM  → drive motor duty
PIN_MOTOR_DIR = 16   # BTS7960 DIR
PIN_SERVO     = 18   # servo signal  (BCM 12/13/18/19 support hardware PWM)
PIN_BUZZER    = 20   # buzzer via NPN driver (2N2222)
PIN_LED       = 21   # warning LED
PIN_RELAY     = 26   # power-cut relay coil (HIGH = energised = actuator power CUT)
PIN_ENC_A     =  5   # encoder channel A  (must be 3.3 V-safe)
PIN_ENC_B     =  6   # encoder channel B  (must be 3.3 V-safe)
PIN_PEDAL     = 19   # digital pedal sensor — manual brake override
PIN_BTN1      = 23   # Button 1: speed select (10 / 20 / 30 km/h)
PIN_BTN2      = 24   # Button 2: IDLE→INIT  |  INIT→RUN
PIN_BTN3      = 25   # Button 3: RESET all states → IDLE

BTN_BOUNCE_MS = 200  # debounce time in ms

ENCODER_PPR           = 20    # pulses per revolution — calibrate to your encoder
WHEEL_CIRCUMFERENCE_M = 0.15  # metres — measure your wheel

# Speed presets for Button 1 (cycles through on each press)
SPEED_PRESETS_MPS = [2.78, 5.56, 8.33]   # 10, 20, 30 km/h
_speed_preset_idx = 0

# Servo calibration — measure with a servo tester or oscilloscope and tune these
SERVO_RELEASE_US  = 1000   # pulse width (µs) when brake fully released
SERVO_PARTIAL_US  = 1500   # pulse width (µs) for partial brake (~40 %)
SERVO_FULL_US     = 2000   # pulse width (µs) for full brake (100 %)
# level 0.0 maps to SERVO_RELEASE_US, level 1.0 maps to SERVO_FULL_US

# Thread-safe event flags set by GPIO interrupt callbacks,
# consumed in the main control loop.
_btn1_event   = threading.Event()
_btn2_event   = threading.Event()
_btn3_event   = threading.Event()
manual_override = False   # True while pedal is pressed

# ============================================================
# GPIO interrupt callbacks  (run in RPi.GPIO callback thread)
# ============================================================
def _pedal_callback(channel):
    global manual_override
    pressed      = GPIO.input(PIN_PEDAL) == GPIO.HIGH
    manual_override = pressed
    set_relay_output(pressed)   # hardware power cut mirrors software override
    if pressed:
        set_brake_output(0.0)   # release brake immediately on override
        print("[PEDAL] Manual override ACTIVE — auto brake disabled", flush=True)
    else:
        print("[PEDAL] Manual override released", flush=True)

def _btn1_callback(channel): _btn1_event.set()
def _btn2_callback(channel): _btn2_event.set()
def _btn3_callback(channel): _btn3_event.set()

# ============================================================
# GPIO setup / teardown
# ============================================================
_motor_pwm = None
_servo_pwm = None   # RPi.GPIO fallback — used only when pigpio unavailable

def _safe_add_event(pin, edge, callback, bouncetime=0):
    """Add edge detection, silently removing any stale registration first."""
    try:
        GPIO.remove_event_detect(pin)
    except Exception:
        pass
    try:
        if bouncetime:
            GPIO.add_event_detect(pin, edge, callback=callback,
                                  bouncetime=bouncetime)
        else:
            GPIO.add_event_detect(pin, edge, callback=callback)
    except RuntimeError as e:
        print(f"[GPIO] Warning: edge detection on pin {pin} failed ({e}) — "
              "callbacks for this pin will not fire", flush=True)

def setup_gpio():
    global _motor_pwm, _servo_pwm
    if not _ON_RPI:
        return
    GPIO.cleanup()           # full kernel reset — clears stale state from crashes
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (PIN_MOTOR_DIR, PIN_BUZZER, PIN_LED, PIN_RELAY):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_MOTOR_PWM, GPIO.OUT)

    GPIO.setup(PIN_ENC_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(PIN_ENC_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    _safe_add_event(PIN_ENC_A, GPIO.RISING, _encoder_isr)

    # Pedal sensor — triggers on both edges so override releases immediately
    GPIO.setup(PIN_PEDAL, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    _safe_add_event(PIN_PEDAL, GPIO.BOTH, _pedal_callback,
                    bouncetime=BTN_BOUNCE_MS)

    # Physical buttons — active LOW (pulled up, button connects to GND)
    for pin, cb in (
        (PIN_BTN1, _btn1_callback),
        (PIN_BTN2, _btn2_callback),
        (PIN_BTN3, _btn3_callback),
    ):
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        _safe_add_event(pin, GPIO.FALLING, cb, bouncetime=BTN_BOUNCE_MS)

    _motor_pwm = GPIO.PWM(PIN_MOTOR_PWM, 1000)   # 1 kHz PWM for motor driver
    _motor_pwm.start(0)

    if _USE_PIGPIO:
        # pigpio: set servo to release position at startup
        _pigpio_pi.set_servo_pulsewidth(PIN_SERVO, SERVO_RELEASE_US)
        print("[GPIO] Servo on pigpio (hardware PWM) — no jitter", flush=True)
    else:
        # RPi.GPIO software PWM fallback — 50 Hz standard servo frequency
        GPIO.setup(PIN_SERVO, GPIO.OUT)
        _servo_pwm = GPIO.PWM(PIN_SERVO, 50)
        release_duty = SERVO_RELEASE_US / 20000.0 * 100.0
        _servo_pwm.start(release_duty)
        print("[GPIO] Servo on RPi.GPIO software PWM (may buzz slightly)", flush=True)

def cleanup_gpio():
    if not _ON_RPI:
        return
    # Park servo at release position before shutdown
    set_servo_position(0.0)
    time.sleep(0.3)
    if _USE_PIGPIO:
        _pigpio_pi.set_servo_pulsewidth(PIN_SERVO, 0)   # 0 = stop sending pulses
        _pigpio_pi.stop()
    elif _servo_pwm:
        _servo_pwm.stop()
    if _motor_pwm:
        _motor_pwm.stop()
    GPIO.cleanup()

# ============================================================
# Encoder — interrupt-driven so no polling lag
# ============================================================
_enc_pulse_count = 0
_enc_lock        = threading.Lock()
_enc_last_count  = 0
_enc_last_time   = time.perf_counter()

def _encoder_isr(channel):
    global _enc_pulse_count
    with _enc_lock:
        _enc_pulse_count += 1

def read_wheel_speed_mps() -> float:
    global _enc_last_count, _enc_last_time
    if not _ON_RPI or USE_MANUAL_SPEED:
        return max(manual_speed_mps, 0.0)
    now = time.perf_counter()
    dt  = now - _enc_last_time
    if dt < 0.005:   # ignore if less than 5 ms since last read
        return 0.0
    with _enc_lock:
        count = _enc_pulse_count
    pulses          = count - _enc_last_count
    _enc_last_count = count
    _enc_last_time  = now
    revs  = pulses / ENCODER_PPR
    speed = (revs * WHEEL_CIRCUMFERENCE_M) / dt
    return float(np.clip(speed, 0.0, 20.0))   # sanity cap at 72 km/h

# ============================================================
# Actuator outputs
# ============================================================
def set_relay_output(energize: bool):
    """
    energize=True  → relay coil ON  → NC opens → actuator 12 V CUT (manual override)
    energize=False → relay coil OFF → NC closed → actuator 12 V ON  (normal operation)
    """
    if not _ON_RPI:
        return
    GPIO.output(PIN_RELAY, GPIO.HIGH if energize else GPIO.LOW)

def set_warning_output(enabled: bool):
    if not _ON_RPI:
        return
    state = GPIO.HIGH if enabled else GPIO.LOW
    GPIO.output(PIN_LED,    state)
    GPIO.output(PIN_BUZZER, state)

def set_brake_output(level: float):
    level = float(np.clip(level, 0.0, 1.0))
    set_servo_position(level)          # servo is the primary brake actuator
    if not _ON_RPI or _motor_pwm is None:
        return
    GPIO.output(PIN_MOTOR_DIR, GPIO.HIGH)
    _motor_pwm.ChangeDutyCycle(level * 100.0)

def set_servo_position(level: float):
    """
    Move the servo to the position corresponding to brake level.

    level 0.0 → SERVO_RELEASE_US  (brake fully off)
    level 1.0 → SERVO_FULL_US     (brake fully on)

    Tune SERVO_RELEASE_US / SERVO_FULL_US in the config above to match
    your physical brake linkage — measure with a servo tester first.
    """
    level    = float(np.clip(level, 0.0, 1.0))
    pulse_us = int(SERVO_RELEASE_US + level * (SERVO_FULL_US - SERVO_RELEASE_US))
    if not _ON_RPI:
        return
    if _USE_PIGPIO and _pigpio_pi is not None:
        _pigpio_pi.set_servo_pulsewidth(PIN_SERVO, pulse_us)
    elif _servo_pwm is not None:
        # duty cycle (%) = pulse_us / period_us * 100, period = 20 000 µs at 50 Hz
        _servo_pwm.ChangeDutyCycle(pulse_us / 20000.0 * 100.0)

# ============================================================
# Vision thread — YOLO runs here, never blocks the control loop
# ============================================================
class _VisionState:
    def __init__(self):
        self.lock     = threading.Lock()
        self.distance = None   # latest distance in metres (None until first detection)
        self.frame    = None   # latest annotated BGR frame for display
        self.running  = True

_vis = _VisionState()

def _vision_worker():
    for frame, tracks in vision_main(yield_every_frame=True):
        if not _vis.running:
            break
        dist = _get_closest_live_distance(tracks)
        with _vis.lock:
            _vis.distance = dist
            _vis.frame    = frame   # reference swap — O(1), no copy

# ============================================================
# Helpers
# ============================================================
def depth_to_meters(depth_value: float) -> float:
    # smoothed_depth from tracker is already in metres after calibration
    return max(float(depth_value), 0.0)

def _get_closest_live_distance(tracks) -> float | None:
    closest = None
    for _, tr in tracks.items():
        depth = getattr(tr, "smoothed_depth", None)
        if depth is None:
            continue
        try:
            depth = float(depth)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(depth):
            continue
        d_m = depth_to_meters(depth)
        if d_m < MIN_VALID_DISTANCE:
            continue
        if closest is None or d_m < closest:
            closest = d_m
    return closest

def ttc_from(distance_m: float, speed_mps: float) -> float:
    if speed_mps <= 1e-6:
        return math.inf
    return distance_m / speed_mps

def stopping_distance(speed_mps: float, decel_mps2: float) -> float:
    if speed_mps <= 1e-6 or decel_mps2 <= 1e-6:
        return 0.0
    return (speed_mps ** 2) / (2.0 * decel_mps2)

def ttc_status(ttc: float) -> str:
    if not math.isfinite(ttc) or ttc > SAFE_TTC:
        return "SAFE"
    if ttc > FCW_TTC:
        return "FCW"
    if ttc > PARTIAL_TTC:
        return "PARTIAL"
    return "EMERGENCY"

def reset_ego(ego: EgoVehicle):
    if hasattr(ego, "reset") and callable(getattr(ego, "reset")):
        ego.reset()
    else:
        for attr in ("position", "z", "velocity"):
            if hasattr(ego, attr):
                setattr(ego, attr, 0.0)
    ego.set_speed(0.0)

def first_state_idx(state_log, target):
    for i, s in enumerate(state_log):
        if s == target:
            return i
    return None

def _resize_letterbox(img, target_w, target_h, bg=(0, 0, 0)):
    """Resize an image without distortion and pad it to the target size."""
    canvas = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
    if img is None or img.size == 0:
        cv2.putText(canvas, "NO CAMERA FRAME", (40, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        return canvas

    h, w = img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _status_color(s):
    if s in ("CRASH", "CRASH_RISK", "EMERGENCY"):
        return (40, 40, 230)
    if s == "PARTIAL":
        return (0, 180, 255)
    if s == "FCW":
        return (0, 230, 230)
    if s == "STOP":
        return (60, 220, 60)
    return (80, 220, 80)


def _draw_label_value(img, label, value, x, y, value_color=(255, 255, 255)):
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 170), 1)
    cv2.putText(img, value, (x, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, value_color, 2)


def _draw_status_panel(panel):
    panel[:] = (0, 0, 0)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (45, 45, 45), 1)

    st = globals().get("state", "IDLE")
    spd = float(globals().get("current_speed", 0.0) or 0.0)
    dist = globals().get("virtual_distance", None)
    live = globals().get("last_live_distance", None)
    ttc_val = globals().get("ttc", math.inf)
    brake = bool(globals().get("brake_on", False))
    warn = bool(globals().get("warning_on", False))
    blevel = float(globals().get("brake_level", 0.0) or 0.0)
    override = bool(globals().get("manual_override", False))
    d0 = globals().get("locked_initial_distance", None)

    col = _status_color(st)
    cv2.putText(panel, "AEB DEMO DASHBOARD", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2)
    cv2.rectangle(panel, (30, 82), (panel.shape[1] - 30, 160), col, -1)
    cv2.putText(panel, f"STATE: {st}", (50, 133), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 0), 3)

    ttc_text = f"{ttc_val:.2f} s" if math.isfinite(ttc_val) else "inf"
    dist_text = f"{float(dist):.2f} m" if dist is not None else "n/a"
    live_text = f"{float(live):.2f} m" if live is not None else "none"
    d0_text = f"{float(d0):.2f} m" if d0 is not None else "not locked"

    _draw_label_value(panel, "Speed", f"{spd:.2f} m/s  ({spd*3.6:.0f} km/h)", 35, 220)
    _draw_label_value(panel, "Virtual distance", dist_text, 35, 315)
    _draw_label_value(panel, "TTC", ttc_text, 35, 410, col)
    _draw_label_value(panel, "Camera distance", live_text, 35, 505)
    _draw_label_value(panel, "Initial lock", d0_text, 35, 600)

    # Brake bar
    bx, by, bw, bh = 360, 260, 260, 42
    cv2.putText(panel, "Brake level", (bx, by - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 170), 1)
    cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (45, 45, 45), 1)
    fill_w = int(np.clip(blevel, 0.0, 1.0) * bw)
    cv2.rectangle(panel, (bx, by), (bx + fill_w, by + bh), col, -1)
    cv2.putText(panel, f"{blevel*100:.0f}%", (bx + 85, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    warn_txt = "WARNING ON" if warn else "warning off"
    brake_txt = "BRAKE ON" if brake else "brake off"
    cv2.putText(panel, warn_txt, (360, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                (0, 230, 230) if warn else (120, 120, 120), 2)
    cv2.putText(panel, brake_txt, (360, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                col if brake else (120, 120, 120), 2)
    if override:
        cv2.rectangle(panel, (350, 470), (panel.shape[1] - 35, 545), (0, 0, 230), -1)
        cv2.putText(panel, "MANUAL OVERRIDE", (370, 518), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    cv2.putText(panel, "Keys: I init/run   R reset   Q quit   W/S speed", (35, panel.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def _draw_live_graph(panel):
    """Draw only Speed vs Time on the dashboard, with clear units."""
    panel[:] = (0, 0, 0)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (45, 45, 45), 1)
    cv2.putText(panel, "LIVE GRAPH: Speed vs Time", (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    times = list(globals().get("time_log", []))
    speeds = list(globals().get("speed_log", []))

    if len(times) < 2 or len(speeds) < 2:
        cv2.putText(panel, "Press I after camera detection to start logging", (35, panel.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        cv2.putText(panel, "X-axis: Time (s)    Y-axis: Speed (m/s)", (35, panel.shape[0] // 2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 180, 180), 2)
        return

    # Keep only the latest GRAPH_HISTORY_SEC seconds for a moving graph.
    now = times[-1]
    start = max(0.0, now - GRAPH_HISTORY_SEC)
    idx0 = 0
    for i, t in enumerate(times):
        if t >= start:
            idx0 = i
            break
    times = times[idx0:]
    speeds = speeds[idx0:]

    # Plot area.
    x0, y0 = 90, 72
    x1, y1 = panel.shape[1] - 45, panel.shape[0] - 65
    cv2.rectangle(panel, (x0, y0), (x1, y1), (45, 45, 45), 1)

    # Dynamic vertical scale, but never lower than the demo maximum so the graph is stable.
    valid_speeds = [float(s) for s in speeds if s is not None and np.isfinite(s)]
    max_speed = max(valid_speeds + [MAX_DEMO_SPEED, 1.0])
    max_speed = math.ceil(max_speed * 1.15)
    max_speed = max(max_speed, 1.0)

    # Grid + Y-axis tick labels in m/s.
    y_ticks = 5
    for k in range(y_ticks + 1):
        y = int(y1 - (y1 - y0) * k / y_ticks)
        speed_val = max_speed * k / y_ticks
        cv2.line(panel, (x0, y), (x1, y), (45, 45, 45), 1)
        cv2.putText(panel, f"{speed_val:.1f}", (25, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Grid + X-axis tick labels in seconds.
    t_min, t_max = times[0], max(times[-1], times[0] + 1e-6)
    x_ticks = 6
    for k in range(x_ticks + 1):
        x = int(x0 + (x1 - x0) * k / x_ticks)
        t_val = t_min + (t_max - t_min) * k / x_ticks
        cv2.line(panel, (x, y0), (x, y1), (45, 45, 45), 1)
        cv2.putText(panel, f"{t_val:.1f}", (x - 22, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Build Speed vs Time polyline.
    pts = []
    for t, v in zip(times, speeds):
        if v is None or not np.isfinite(v):
            continue
        x = int(x0 + (float(t) - t_min) / (t_max - t_min) * (x1 - x0))
        y = int(y1 - np.clip(float(v) / max_speed, 0.0, 1.0) * (y1 - y0))
        pts.append((x, y))

    if len(pts) >= 2:
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (80, 255, 80), 3)

    # Current speed label.
    current_spd = float(globals().get("current_speed", 0.0) or 0.0)
    cv2.circle(panel, pts[-1], 5, (80, 255, 80), -1) if pts else None
    cv2.putText(panel, f"Current speed: {current_spd:.2f} m/s  ({current_spd * 3.6:.1f} km/h)",
                (x0 + 20, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 255, 80), 2)

    # Axis labels with units.
    cv2.putText(panel, "Time (s)", ((x0 + x1) // 2 - 55, panel.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(panel, "Speed (m/s)", (x0 + 10, y0 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
    cv2.putText(panel, f"Showing last {GRAPH_HISTORY_SEC:.0f} s", (x1 - 185, y0 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)


def _draw_ego_animation(panel):
    panel[:] = (0, 0, 0)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (45, 45, 45), 1)
    cv2.putText(panel, "EGO VEHICLE ANIMATION", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    st = globals().get("state", "IDLE")
    locked = globals().get("locked_initial_distance", None)
    robot = float(globals().get("robot_z", 0.0) or 0.0)
    dist = globals().get("virtual_distance", None)
    brake = bool(globals().get("brake_on", False))

    road_y = panel.shape[0] // 2 + 45
    cv2.rectangle(panel, (30, road_y - 55), (panel.shape[1] - 30, road_y + 55), (18, 18, 18), -1)
    cv2.line(panel, (30, road_y), (panel.shape[1] - 30, road_y), (170, 170, 170), 2)
    for x in range(40, panel.shape[1] - 30, 80):
        cv2.line(panel, (x, road_y), (x + 35, road_y), (230, 230, 230), 2)

    span = max(float(locked) if locked else 5.0, 5.0)
    left, right = 80, panel.shape[1] - 95
    obj_x = right
    car_x = int(left + np.clip(robot / span, 0.0, 1.0) * (right - left))

    # obstacle
    cv2.rectangle(panel, (obj_x - 8, road_y - 70), (obj_x + 30, road_y + 45), (0, 0, 220), -1)
    cv2.putText(panel, "OBSTACLE", (obj_x - 75, road_y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    # ego car body
    car_color = _status_color(st)
    cv2.rectangle(panel, (car_x - 50, road_y - 32), (car_x + 50, road_y + 28), car_color, -1)
    cv2.rectangle(panel, (car_x - 22, road_y - 58), (car_x + 35, road_y - 28), (210, 210, 210), -1)
    cv2.circle(panel, (car_x - 32, road_y + 32), 12, (10, 10, 10), -1)
    cv2.circle(panel, (car_x + 35, road_y + 32), 12, (10, 10, 10), -1)
    cv2.putText(panel, "EGO", (car_x - 28, road_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    if brake:
        cv2.putText(panel, "BRAKING", (car_x - 55, road_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 230), 2)
        cv2.line(panel, (car_x - 60, road_y + 50), (car_x - 115, road_y + 50), (0, 230, 230), 3)
        cv2.line(panel, (car_x - 55, road_y + 66), (car_x - 100, road_y + 66), (0, 230, 230), 2)

    d_text = f"Remaining: {float(dist):.2f} m" if dist is not None else "Remaining: n/a"
    cv2.putText(panel, d_text, (35, panel.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, f"Travel: {robot:.2f} m", (35, panel.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 1)



def _exit_button_rect():
    x1 = SCREEN_W - EXIT_BUTTON_MARGIN - EXIT_BUTTON_W
    y1 = EXIT_BUTTON_MARGIN
    x2 = x1 + EXIT_BUTTON_W
    y2 = y1 + EXIT_BUTTON_H
    return x1, y1, x2, y2


def _draw_exit_button(img):
    x1, y1, x2, y2 = _exit_button_rect()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(img, "X", (x1 + 18, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.35, (255, 255, 255), 3)
    cv2.putText(img, "EXIT", (x1 + 62, y1 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def _mouse_callback(event, x, y, flags, param):
    global _exit_requested
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x1, y1, x2, y2 = _exit_button_rect()
    if x1 <= x <= x2 and y1 <= y <= y2:
        print("[MOUSE] EXIT button clicked — quitting", flush=True)
        _exit_requested = True

def _make_dashboard(cam_frame):
    dashboard = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    camera_panel = _resize_letterbox(cam_frame, CAM_W, TOP_H)
    status_panel = np.zeros((TOP_H, STATUS_W, 3), dtype=np.uint8)
    graph_panel = np.zeros((BOTTOM_H, GRAPH_W, 3), dtype=np.uint8)
    ego_panel = np.zeros((BOTTOM_H, EGO_W, 3), dtype=np.uint8)

    _draw_status_panel(status_panel)
    _draw_live_graph(graph_panel)
    _draw_ego_animation(ego_panel)

    dashboard[0:TOP_H, 0:CAM_W] = camera_panel
    dashboard[0:TOP_H, CAM_W:SCREEN_W] = status_panel
    dashboard[TOP_H:SCREEN_H, 0:GRAPH_W] = graph_panel
    dashboard[TOP_H:SCREEN_H, GRAPH_W:SCREEN_W] = ego_panel

    # separators
    cv2.line(dashboard, (CAM_W, 0), (CAM_W, TOP_H), (45, 45, 45), 1)
    cv2.line(dashboard, (0, TOP_H), (SCREEN_W, TOP_H), (45, 45, 45), 1)
    cv2.line(dashboard, (GRAPH_W, TOP_H), (GRAPH_W, SCREEN_H), (45, 45, 45), 1)

    # Draw this last so it stays clickable and visible above all panels.
    _draw_exit_button(dashboard)
    return dashboard


def _show(cam_frame, world_img=None):
    """
    Display output.

    TV_MODE=True + DASHBOARD_MODE=True creates one 1920x1080 HDMI
    dashboard for the Sharp 2T-C42BE1:
      top-left: camera feedback
      top-right: large status panel
      bottom-left: moving live graph
      bottom-right: moving ego-car animation
    """
    if TV_MODE:
        if DASHBOARD_MODE:
            cv2.imshow("AEB System", _make_dashboard(cam_frame))
            return

        # Legacy TV layout: camera on top, world panel below.
        if cam_frame is None or world_img is None:
            return
        cam_h = int(np.clip(CAM_DISPLAY_H, 1, SCREEN_H - 1))
        world_h = SCREEN_H - cam_h
        cam_resized = cv2.resize(cam_frame, (SCREEN_W, cam_h), interpolation=cv2.INTER_LINEAR)
        world_resized = cv2.resize(world_img, (SCREEN_W, world_h), interpolation=cv2.INTER_LINEAR)
        combined = np.vstack([cam_resized, world_resized])
        cv2.imshow("AEB System", combined)
    else:
        cv2.imshow("CV + Tracking", cam_frame)
        cv2.imshow("2D World", world_img)

def _ms(s: float) -> float:
    return s * 1000.0

def print_timing(label: str, timings: dict):
    if not DEBUG_TIMING:
        return
    parts = " | ".join(f"{k}={_ms(v):.2f}ms" for k, v in timings.items())
    print(f"[{label}] {parts} | total={_ms(sum(timings.values())):.2f}ms", flush=True)

# ============================================================
# Plot
# ============================================================
def plot_results(time_log, distance_log, speed_log, ttc_log,
                 travel_log, stop_req_log, state_log):
    # Disabled during HDMI dashboard demo because matplotlib plt.show() can steal focus
    # or block the OpenCV window, making it feel impossible to close.
    if not SHOW_MATPLOTLIB_PLOTS or not time_log:
        return

    fcw_idx       = first_state_idx(state_log, "FCW")
    partial_idx   = first_state_idx(state_log, "PARTIAL")
    emergency_idx = first_state_idx(state_log, "EMERGENCY")
    stop_idx      = first_state_idx(state_log, "STOP")
    crash_idx     = first_state_idx(state_log, "CRASH")

    markers = [
        (fcw_idx,       "FCW trigger"),
        (partial_idx,   "Partial brake"),
        (emergency_idx, "Emergency brake"),
        (stop_idx,      "Stop"),
        (crash_idx,     "Crash"),
    ]

    def _vlines():
        for idx, lbl in markers:
            if idx is not None:
                plt.axvline(time_log[idx], linestyle="--", label=lbl)

    plt.figure(figsize=(9, 4))
    plt.plot(time_log, distance_log, label="Remaining distance")
    _vlines()
    plt.xlabel("Time (s)"); plt.ylabel("Distance (m)")
    plt.title("Distance vs Time"); plt.grid(True); plt.legend()

    plt.figure(figsize=(9, 4))
    plt.plot(time_log, speed_log, label="Speed")
    _vlines()
    plt.xlabel("Time (s)"); plt.ylabel("Speed (m/s)")
    plt.title("Speed vs Time"); plt.grid(True); plt.legend()

    plt.figure(figsize=(9, 4))
    ttc_arr    = np.array(ttc_log, dtype=float)
    finite_ttc = ttc_arr[np.isfinite(ttc_arr)]
    y_top = max(float(np.max(finite_ttc)), SAFE_TTC + 0.5) if finite_ttc.size > 0 else SAFE_TTC + 1.0
    plt.axhspan(0,           PARTIAL_TTC, color="red",    alpha=0.20, label="Emergency")
    plt.axhspan(PARTIAL_TTC, FCW_TTC,     color="orange", alpha=0.20, label="Partial")
    plt.axhspan(FCW_TTC,     SAFE_TTC,    color="yellow", alpha=0.20, label="FCW")
    plt.axhspan(SAFE_TTC,    y_top,       color="green",  alpha=0.12, label="Safe")
    for thresh in (SAFE_TTC, FCW_TTC, PARTIAL_TTC):
        plt.axhline(thresh, linestyle="--", color="blue")
    plt.plot(time_log, ttc_log, color="black", label="TTC")
    _vlines()
    plt.xlabel("Time (s)"); plt.ylabel("TTC (s)")
    plt.title("TTC vs Time"); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# State variables
# ============================================================
ego = EgoVehicle()
ego.set_speed(0.0)

state = "IDLE"

locked_initial_distance  = None
virtual_distance         = None
robot_z                  = 0.0

init_samples    = []
init_start_time = None

brake_on    = False
warning_on  = False
brake_level = 0.0
plots_shown = False

brake_trigger_speed          = 0.0
brake_required_stop_distance = 0.0
brake_trigger_mode           = None

last_live_distance = None

time_log     = []
distance_log = []
speed_log    = []
ttc_log      = []
travel_log   = []
stop_req_log = []
state_log    = []
sim_time     = 0.0

current_speed = 0.0
ttc           = math.inf
status        = "SAFE"

# ============================================================
# Startup
# ============================================================
setup_gpio()

if TV_MODE:
    cv2.namedWindow("AEB System", cv2.WINDOW_NORMAL)
    cv2.moveWindow("AEB System", 0, 0)
    cv2.resizeWindow("AEB System", SCREEN_W, SCREEN_H)
    cv2.setMouseCallback("AEB System", _mouse_callback)
    if USE_OPENCV_FULLSCREEN:
        # Warning: this can crash on some Raspberry Pi/OpenCV Qt builds with:
        # FATAL: exception not rethrown
        cv2.setWindowProperty("AEB System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

_vision_thread = threading.Thread(target=_vision_worker, daemon=True)
_vision_thread.start()

print("Waiting for camera...", flush=True)
while True:
    with _vis.lock:
        ready = _vis.frame is not None
    if ready:
        break
    time.sleep(0.05)
print("Camera ready. Starting control loop at 50 Hz.", flush=True)

# ============================================================
# Control loop — runs at CONTROL_HZ regardless of YOLO speed
# ============================================================
loop_counter = 0

try:
    while True:
        loop_t0      = time.perf_counter()
        loop_counter += 1
        should_display = (loop_counter % DISPLAY_EVERY_N == 0)
        timings = {}

        # ----------------------------------------------------
        # Snapshot latest vision data — non-blocking
        # ----------------------------------------------------
        t0 = time.perf_counter()
        with _vis.lock:
            live_distance = _vis.distance
            cam_frame     = _vis.frame   # shared ref; vision thread never mutates in place
        last_live_distance = live_distance if live_distance is not None else last_live_distance
        timings["vision"] = time.perf_counter() - t0

        # ----------------------------------------------------
        # Keyboard input + display — throttled to ~17 FPS
        # ----------------------------------------------------
        t0  = time.perf_counter()
        key = cv2.waitKey(1) & 0xFF  # always poll keyboard so q/ESC works even between display frames

        # Physical button events override the keyboard key for this tick
        if _btn1_event.is_set():
            _btn1_event.clear()
            _speed_preset_idx = (_speed_preset_idx + 1) % len(SPEED_PRESETS_MPS)
            manual_speed_mps  = SPEED_PRESETS_MPS[_speed_preset_idx]
            print(f"[BTN1] Speed → {manual_speed_mps:.2f} m/s "
                  f"({manual_speed_mps * 3.6:.0f} km/h)", flush=True)
        if _btn2_event.is_set():
            _btn2_event.clear()
            key = ord("i")
        if _btn3_event.is_set():
            _btn3_event.clear()
            key = ord("r")

        # Exit keys: q/Q or ESC. Also exit if the user closes the OpenCV window.
        if key in (ord("q"), ord("Q"), 27) or _exit_requested:
            break
        if TV_MODE:
            try:
                if cv2.getWindowProperty("AEB System", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        # Start/init keys: accept lowercase i and uppercase I.
        # IMPORTANT: this must be a separate if, not an elif attached to the TV_MODE check.
        if key in (ord("i"), ord("I")) and state in ("IDLE", "STOP", "CRASH"):
            print("[KEY] INIT pressed — starting initialization", flush=True)
            state           = "INIT"
            init_samples    = []
            init_start_time = time.perf_counter()
            locked_initial_distance  = None
            virtual_distance         = None
            robot_z                  = 0.0
            brake_on    = False; warning_on  = False; brake_level = 0.0
            plots_shown = False
            brake_trigger_speed          = 0.0
            brake_required_stop_distance = 0.0
            brake_trigger_mode           = None
            time_log.clear();  distance_log.clear(); speed_log.clear()
            ttc_log.clear();   travel_log.clear();   stop_req_log.clear()
            state_log.clear()
            sim_time = 0.0; current_speed = 0.0
            set_warning_output(False); set_brake_output(0.0)
            reset_ego(ego)

        elif key in (ord("r"), ord("R")):
            print("[KEY] RESET pressed — returning to IDLE", flush=True)
            state           = "IDLE"
            init_samples    = []
            init_start_time = None
            locked_initial_distance  = None
            virtual_distance         = None
            robot_z                  = 0.0
            brake_on    = False; warning_on  = False; brake_level = 0.0
            plots_shown = False
            brake_trigger_speed          = 0.0
            brake_required_stop_distance = 0.0
            brake_trigger_mode           = None
            time_log.clear();  distance_log.clear(); speed_log.clear()
            ttc_log.clear();   travel_log.clear();   stop_req_log.clear()
            state_log.clear()
            sim_time = 0.0; current_speed = 0.0
            set_warning_output(False); set_brake_output(0.0)
            reset_ego(ego)

        elif USE_MANUAL_SPEED:
            if   key in (ord("w"), ord("W")): manual_speed_mps = min(manual_speed_mps + MANUAL_SPEED_STEP, MAX_DEMO_SPEED)
            elif key in (ord("s"), ord("S")): manual_speed_mps = max(manual_speed_mps - MANUAL_SPEED_STEP, 0.0)
            elif key == ord(" "): manual_speed_mps = 0.0

        timings["key"] = time.perf_counter() - t0

        # ----------------------------------------------------
        # IDLE
        # ----------------------------------------------------
        if state == "IDLE":
            t0    = time.perf_counter()
            world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)
            cv2.putText(world, "IDLE - press I to initialize",
                        (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(world, f"Manual speed: {manual_speed_mps:.2f} m/s",
                        (20, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(world, "W/S adjust speed, SPACE zeroes it",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            dist_str = f"{last_live_distance:.2f} m" if last_live_distance else "none"
            cv2.putText(world, f"Live camera distance: {dist_str}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            if should_display and cam_frame is not None:
                _show(cam_frame, world)
            timings["render"] = time.perf_counter() - t0

            if loop_counter % PRINT_EVERY_N_LOOPS == 0:
                print_timing("IDLE", timings)
            time.sleep(max(0.0, DT - (time.perf_counter() - loop_t0)))
            continue

        # ----------------------------------------------------
        # INIT
        # ----------------------------------------------------
        if state == "INIT":
            t0 = time.perf_counter()
            if live_distance is not None:
                init_samples.append(live_distance)

            elapsed = (time.perf_counter() - init_start_time) if init_start_time else 0.0
            world   = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)
            cv2.putText(world, "INIT - locking initial distance",
                        (20,  35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(world, f"Samples: {len(init_samples)}/{INIT_REQUIRED_SAMPLES}",
                        (20,  70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255,   0), 2)
            cv2.putText(world, f"Elapsed: {elapsed:.2f} s",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255,   0), 2)
            dist_str = f"{live_distance:.2f} m" if live_distance else "waiting for detection"
            cv2.putText(world, f"Current live distance: {dist_str}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            if (len(init_samples) >= INIT_REQUIRED_SAMPLES) or (
                init_start_time and elapsed >= INIT_MAX_WAIT_SEC and len(init_samples) > 0
            ):
                locked_initial_distance = float(np.mean(init_samples))
                virtual_distance        = locked_initial_distance
                robot_z       = 0.0
                current_speed = 0.0
                brake_on    = False; warning_on  = False; brake_level = 0.0
                set_warning_output(False); set_brake_output(0.0)
                reset_ego(ego); ego.set_speed(0.0)
                state = "RUN"

            if should_display and cam_frame is not None:
                _show(cam_frame, world)
            timings["render"] = time.perf_counter() - t0

            if loop_counter % PRINT_EVERY_N_LOOPS == 0:
                print_timing("INIT", timings)
            time.sleep(max(0.0, DT - (time.perf_counter() - loop_t0)))
            continue

        # ----------------------------------------------------
        # RUN / FCW / PARTIAL / EMERGENCY / STOP / CRASH
        # Runs at CONTROL_HZ — actuators updated every DT = 20 ms
        # ----------------------------------------------------
        t0              = time.perf_counter()
        wheel_speed_mps = read_wheel_speed_mps()
        timings["read_speed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        virtual_distance = (
            max(locked_initial_distance - robot_z, 0.0)
            if locked_initial_distance is not None else 0.0
        )
        timings["dist_calc"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if state in ("RUN", "FCW"):
            current_speed = wheel_speed_mps
            ego.set_speed(current_speed)
            ttc    = ttc_from(virtual_distance, current_speed)
            status = ttc_status(ttc)

            if status == "SAFE":
                state = "RUN"; warning_on = False; brake_on = False; brake_level = 0.0
                set_warning_output(False); set_brake_output(0.0)
                robot_z += current_speed * DT

            elif status == "FCW":
                state = "FCW"; warning_on = True; brake_on = False; brake_level = 0.0
                set_warning_output(True); set_brake_output(0.0)
                robot_z += current_speed * DT

            elif status == "PARTIAL":
                state = "PARTIAL"; warning_on = True; brake_on = not manual_override
                brake_level         = PARTIAL_BRAKE_DECEL / FULL_BRAKE_DECEL
                brake_trigger_speed = current_speed
                brake_trigger_mode  = "PARTIAL"
                brake_required_stop_distance = stopping_distance(current_speed, PARTIAL_BRAKE_DECEL)
                set_warning_output(True)
                if not manual_override:
                    set_brake_output(brake_level)

            else:  # EMERGENCY
                state = "EMERGENCY"; warning_on = True; brake_on = not manual_override
                brake_level = 1.0
                brake_trigger_speed = current_speed
                brake_trigger_mode  = "EMERGENCY"
                brake_required_stop_distance = stopping_distance(current_speed, FULL_BRAKE_DECEL)
                set_warning_output(True)
                if not manual_override:
                    set_brake_output(1.0)

        elif state == "PARTIAL":
            set_warning_output(True)
            if not manual_override:
                set_brake_output(PARTIAL_BRAKE_DECEL / FULL_BRAKE_DECEL)
            current_speed    = max(current_speed - PARTIAL_BRAKE_DECEL * DT, 0.0)
            ego.set_speed(current_speed)
            robot_z         += current_speed * DT
            virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance else 0.0
            ttc              = ttc_from(virtual_distance, current_speed)
            status           = "PARTIAL"
            if virtual_distance <= 0.0 and current_speed > STOP_EPS:
                state = "CRASH"; current_speed = 0.0; ego.set_speed(0.0)
                virtual_distance = 0.0; brake_level = 1.0; set_brake_output(1.0)
            elif current_speed <= STOP_EPS:
                state = "STOP"; current_speed = 0.0; ego.set_speed(0.0)
                brake_on = True; brake_level = PARTIAL_BRAKE_DECEL / FULL_BRAKE_DECEL
                set_brake_output(brake_level)

        elif state == "EMERGENCY":
            set_warning_output(True)
            if not manual_override:
                set_brake_output(1.0)
            current_speed    = max(current_speed - FULL_BRAKE_DECEL * DT, 0.0)
            ego.set_speed(current_speed)
            robot_z         += current_speed * DT
            virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance else 0.0
            ttc              = ttc_from(virtual_distance, current_speed)
            status           = "EMERGENCY"
            if virtual_distance <= 0.0 and current_speed > STOP_EPS:
                state = "CRASH"; current_speed = 0.0; ego.set_speed(0.0)
                virtual_distance = 0.0; brake_level = 1.0; set_brake_output(1.0)
            elif current_speed <= STOP_EPS:
                state = "STOP"; current_speed = 0.0; ego.set_speed(0.0)
                brake_on = True; brake_level = 1.0; set_brake_output(1.0)

        elif state == "STOP":
            current_speed = 0.0; ego.set_speed(0.0)
            warning_on = False; brake_on = True; brake_level = 0.0
            set_warning_output(False); set_brake_output(0.0)
            ttc = math.inf; status = "STOP"

        elif state == "CRASH":
            current_speed = 0.0; ego.set_speed(0.0)
            warning_on = True; brake_on = True; brake_level = 1.0
            set_warning_output(True); set_brake_output(1.0)
            virtual_distance = 0.0; ttc = math.inf; status = "CRASH_RISK"

        timings["logic"] = time.perf_counter() - t0

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        t0 = time.perf_counter()
        sim_time += DT
        time_log.append(sim_time)
        distance_log.append(virtual_distance if virtual_distance is not None else 0.0)
        speed_log.append(current_speed)
        ttc_log.append(ttc if math.isfinite(ttc) else np.nan)
        travel_log.append(robot_z)
        stop_req_log.append(brake_required_stop_distance if state in ("PARTIAL", "EMERGENCY") else 0.0)
        state_log.append(state)
        timings["log"] = time.perf_counter() - t0

        # ----------------------------------------------------
        # Visualization — only on display frames
        # ----------------------------------------------------
        t0 = time.perf_counter()
        if should_display and cam_frame is not None:
            world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

            cv2.putText(world, f"STATE: {state}",
                        (20,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(world, f"Speed: {current_speed:.2f} m/s",
                        (20,  65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if locked_initial_distance is not None:
                cv2.putText(world, f"Initial d0: {locked_initial_distance:.2f} m",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            dist_text = f"{virtual_distance:.2f} m" if virtual_distance is not None else "n/a"
            ttc_text  = f"{ttc:.2f} s" if math.isfinite(ttc) else "inf"
            cv2.putText(world, f"Virtual dist: {dist_text}",
                        (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            req_text = f"{brake_required_stop_distance:.2f} m" if state in ("PARTIAL", "EMERGENCY") else "n/a"
            cv2.putText(world, f"TTC: {ttc_text}   Stop dist: {req_text}",
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(world, f"Brake: {'ON' if brake_on else 'OFF'}   Warning: {'ON' if warning_on else 'OFF'}",
                        (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255,   0), 2)
            override_color = (0, 0, 255) if manual_override else (255, 255, 0)
            override_txt   = "MANUAL OVERRIDE" if manual_override else f"Brake level: {brake_level:.2f}   Mode: {status}"
            cv2.putText(world, override_txt,
                        (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, override_color, 2)

            span_m       = max(locked_initial_distance if locked_initial_distance else 5.0, 5.0)
            display_span = WORLD_WIDTH - 140
            left_x       = 60
            robot_x  = int(left_x + min(max((robot_z / span_m) * display_span, 0.0), display_span))
            object_x = (
                int(left_x + min(max((locked_initial_distance / span_m) * display_span, 0.0), display_span))
                if locked_initial_distance else WORLD_WIDTH - 80
            )
            cv2.rectangle(world,
                          (robot_x,  ROBOT_Y  - BOX_H // 2),
                          (robot_x  + BOX_W, ROBOT_Y  + BOX_H // 2),
                          (0, 255, 0) if state != "CRASH" else (0, 0, 255), -1)
            cv2.rectangle(world,
                          (object_x, OBJECT_Y - BOX_H // 2),
                          (object_x + BOX_W, OBJECT_Y + BOX_H // 2),
                          (0, 0, 255), -1)
            cv2.line(world, (robot_x + BOX_W, ROBOT_Y), (object_x, OBJECT_Y), (255, 255, 255), 2)

            state_labels = {
                "CRASH":     ("COLLISION OCCURRED",        (0,   0, 255)),
                "STOP":      ("STOPPED SAFELY",            (0, 255,   0)),
                "PARTIAL":   ("PARTIAL BRAKING",           (0, 255, 255)),
                "EMERGENCY": ("EMERGENCY BRAKING",         (0,   0, 255)),
                "FCW":       ("FORWARD COLLISION WARNING", (0, 255, 255)),
            }
            if state in state_labels:
                txt, col = state_labels[state]
                cv2.putText(world, txt, (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

            _show(cam_frame, world)
        timings["render"] = time.perf_counter() - t0

        # ----------------------------------------------------
        # Plot once on STOP / CRASH
        # ----------------------------------------------------
        if not plots_shown and state in ("STOP", "CRASH"):
            plots_shown = True
            plot_results(time_log, distance_log, speed_log, ttc_log,
                         travel_log, stop_req_log, state_log)

        if loop_counter % PRINT_EVERY_N_LOOPS == 0:
            print_timing(state, timings)

        # ----------------------------------------------------
        # Sleep to maintain 50 Hz — absorbs any leftover time
        # ----------------------------------------------------
        time.sleep(max(0.0, DT - (time.perf_counter() - loop_t0)))

except KeyboardInterrupt:
    pass

finally:
    _vis.running = False
    cv2.destroyAllWindows()
    cleanup_gpio()
    if time_log and not plots_shown:
        plot_results(time_log, distance_log, speed_log, ttc_log,
                     travel_log, stop_req_log, state_log)
