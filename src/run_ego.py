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
DISPLAY_EVERY_N     = 3     # ~17 FPS display at 50 Hz
TV_MODE             = False  # set True for HDMI TV; one combined fullscreen window

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

def _show(cam_frame, world_img):
    if TV_MODE:
        cam_h, cam_w = cam_frame.shape[:2]
        w_h          = cam_w * WORLD_HEIGHT // WORLD_WIDTH
        world_scaled = cv2.resize(world_img, (cam_w, w_h))
        combined     = np.vstack([cam_frame, world_scaled])
        cv2.imshow("AEB System", combined)
    else:
        cv2.imshow("CV + Tracking", cam_frame)
        cv2.imshow("2D World",      world_img)

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
    if not time_log:
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
        key = (cv2.waitKey(1) & 0xFF) if should_display else 0xFF

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

        if key == ord("q"):
            break

        elif key == ord("i") and state in ("IDLE", "STOP", "CRASH"):
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

        elif key == ord("r"):
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
            if   key == ord("w"): manual_speed_mps = min(manual_speed_mps + MANUAL_SPEED_STEP, MAX_DEMO_SPEED)
            elif key == ord("s"): manual_speed_mps = max(manual_speed_mps - MANUAL_SPEED_STEP, 0.0)
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
