#!/usr/bin/env python3
"""
run_ego_raspi_arduino_full.py

FINAL Raspberry Pi side for the AEB demonstrator using:

Raspberry Pi:
- camera / YOLO distance
- TTC / AEB state machine
- display
- servo brake
- relay output
- warning LED / buzzer
- physical buttons
- USB serial communication with Arduino

Arduino:
- MDD10 PWM/DIR drive motor control
- encoder A/B speed reading
- analog pedal A0 reading
- motor safety stop when analog pedal is pressed
- telemetry back to Raspberry Pi

Expected Arduino telemetry format:
    T <rpm> <speed_mps> <pulse_count> <direction> <pedal_percent> <pedal_pressed> <pedal_adc> <pedal_voltage>

Example:
    T 398.50 0.996 1327 1 52.4 1 529 2.59

Expected Arduino motor command:
    M <pwm> <dir>

Example:
    M 120 1

Important:
- Raspberry Pi does NOT directly control MDD10 PWM/DIR.
- Raspberry Pi does NOT read analog pedal directly.
- Raspberry Pi reads encoder speed and pedal state from Arduino serial telemetry.
- Servo brake and drive motor are separated.
"""

import cv2
import numpy as np
import time
import math
import threading
import serial
import serial.tools.list_ports
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no Qt/xcb needed
import matplotlib.pyplot as plt

from ego_sim import EgoVehicle
from webcam_distance_test import main as vision_main


# ============================================================
# RPi GPIO
# ============================================================
try:
    import RPi.GPIO as GPIO
    _ON_RPI = True
except ImportError:
    GPIO = None
    _ON_RPI = False


# ============================================================
# pigpio for stable servo PWM
# ============================================================
try:
    import pigpio as _pigpio_mod
    _pigpio_pi = _pigpio_mod.pi() if _ON_RPI else None
    _USE_PIGPIO = _ON_RPI and (_pigpio_pi is not None) and _pigpio_pi.connected
except Exception:
    _pigpio_pi = None
    _USE_PIGPIO = False


# ============================================================
# Arduino serial config
# ============================================================
USE_ARDUINO = True
ARDUINO_PORT = "AUTO"
ARDUINO_BAUD = 115200
ARDUINO_TIMEOUT_SEC = 0.02

# For camera/display debugging only. For real demo, set this False.
ALLOW_RUN_WITHOUT_ARDUINO = True

# If encoder telemetry disappears for too long, force speed to zero.
ARDUINO_TELEMETRY_STALE_SEC = 1.0


# ============================================================
# Timing / display config
# ============================================================
DEBUG_TIMING = True
PRINT_EVERY_N_LOOPS = 150
DISPLAY_EVERY_N = 3
TV_MODE = True
DASHBOARD_MODE = True
SCREEN_W = 1920
SCREEN_H = 1080
TOP_H = int(SCREEN_H * 0.65)
BOTTOM_H = SCREEN_H - TOP_H
CAM_W = int(SCREEN_W * 0.60)
STATUS_W = SCREEN_W - CAM_W
GRAPH_W = int(SCREEN_W * 0.55)
EGO_W = SCREEN_W - GRAPH_W
GRAPH_HISTORY_SEC = 15.0
EXIT_BUTTON_W = 120
EXIT_BUTTON_H = 50
EXIT_BUTTON_MARGIN = 20

CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ


# ============================================================
# World / display geometry
# ============================================================
WORLD_WIDTH = 1000
WORLD_HEIGHT = 260
ROBOT_Y = WORLD_HEIGHT // 2
OBJECT_Y = WORLD_HEIGHT // 2
BOX_W = 40
BOX_H = 40

MIN_VALID_DISTANCE = 0.10


# ============================================================
# Init behaviour
# ============================================================
INIT_REQUIRED_SAMPLES = 8
INIT_MAX_WAIT_SEC = 2.0


# ============================================================
# TTC / braking thresholds
# ============================================================
SAFE_TTC = 2.6
FCW_TTC = 1.6
PARTIAL_TTC = 0.6

FULL_BRAKE_DECEL = 6.43
PARTIAL_BRAKE_DECEL = FULL_BRAKE_DECEL * 0.4

STOP_EPS = 0.01


# ============================================================
# Demo speed command
# ============================================================
# This is only for requested motor PWM command.
# Actual measured speed comes from Arduino encoder telemetry.
MANUAL_SPEED_STEP = 0.10
MAX_DEMO_SPEED = 5.0
manual_speed_mps = 0.0

MAX_MOTOR_PWM = 255


# ============================================================
# GPIO pin assignments on Raspberry Pi
# ============================================================
# MDD10 PWM/DIR are controlled by Arduino, not Pi.
PIN_SERVO = 18
PIN_BUZZER = 16
PIN_LED = 21
PIN_RELAY = 26       # HIGH = relay coil energised = power CUT, if wired that way

PIN_BTN1 = 5         # speed select
PIN_BTN2 = 6         # IDLE/STOP/CRASH -> INIT
PIN_BTN3 = 25        # RESET -> IDLE

# Optional: if you wired Arduino D7 open-drain pedal output to Pi GPIO.
# Default False because serial telemetry already gives pedal state.
USE_PEDAL_GPIO_FROM_ARDUINO = False
PIN_PEDAL_FROM_ARDUINO = 19   # Arduino D7 open-drain -> Pi GPIO19 with pull-up
PEDAL_GPIO_ACTIVE_LOW = True  # D7 pulls LOW when pedal pressed

BTN_BOUNCE_MS = 200

SPEED_PRESETS_MPS = [2.78, 5.56, 8.33]  # 10, 20, 30 km/h
SPEED_PRESETS_PWM = [51,   96,   140]   # measured open-loop PWM for each speed
_speed_preset_idx = len(SPEED_PRESETS_MPS) - 1

SERVO_RELEASE_US        = 2000
SERVO_FULL_US           = 1000
SERVO_AEB_RELEASE_DEG   = 60    # rest / no-brake position (physical 0 of brake travel)
SERVO_AEB_PARTIAL_DEG   = 70    # partial brake (PARTIAL state)
SERVO_AEB_FULL_DEG      = 85    # full brake (EMERGENCY state)
PEDAL_FULL_PERCENT      = 100.0  # pedal % = full servo travel (recalibrated Arduino, pedal now reads 0–100%)

# DS3230 servo (270° version) physical parameters for system latency calculation.
# Full pulse range: 500–2500µs = 270° physical.
# Speed at 5V: 0.20 s/60° → 300 deg/s.
# τ_mechanical = (Δpulse_us / SERVO_PHYS_PULSE_RANGE_US) × SERVO_PHYS_ANGLE_DEG / SERVO_SPEED_DEG_PER_S
SERVO_PHYS_PULSE_RANGE_US = 2000.0   # 2500 - 500 µs spans 270°
SERVO_PHYS_ANGLE_DEG      = 270.0
SERVO_SPEED_DEG_PER_S     = 300.0    # DS3230 at 5V

_latency_log: list = []              # records per AEB trigger for CSV export
_pending_latency_t: float | None = None  # set just before each AEB servo command

_btn_state = [False, False, False]   # True = currently pressed, for edge detection


# ============================================================
# Runtime input state from Arduino
# ============================================================
manual_override = False       # True when analog pedal is pressed
arduino_speed_mps = 0.0
arduino_rpm = 0.0

arduino_pulse_count = 0
arduino_direction = 1
pedal_percent = 0.0
pedal_pressed = False
pedal_adc = 0
pedal_voltage = 0.0
last_telemetry_time = 0.0


# ============================================================
# Module-level display state (updated each render loop)
# ============================================================
_exit_requested = False
_ds: dict = {}
_relay_open_after: float | None = None

# ── RPM closed-loop controller (calibrated against test_motor_relay.py) ──
WHEEL_CIRCUMFERENCE_M  = 1.596   # 20-inch wheel: pi × (20 × 0.0254)

_RPM_KP               = 0.02    # P-gain: PWM units per RPM of error per telemetry tick
_RPM_MIN_RUN_PWM      = 50      # floor: prevents motor stalling into dead zone
_RPM_SMOOTH_N         = 4       # rolling-average window (4 × 100 ms = 400 ms) — less lag = less oscillation
_RPM_BOOST_PWM        = 200     # startup PWM — avoids 100% duty-cycle ADC noise
_RPM_BOOST_DURATION_S = 1.5     # max boost seconds — exits early if target speed reached

_rpm_buf: list  = []
_enc_rpm_smoothed     = 0.0
_rpm_target           = 0.0     # target RPM for closed-loop
_rpm_pwm_actual       = 0       # PWM currently being sent
_rpm_boost_end        = 0.0
_rpm_last_enc_t       = 0.0     # perf_counter of last received T line
_rpm_prev_enc_t       = 0.0     # detect new telemetry packet in tick


# ============================================================
# Arduino serial driver
# ============================================================
class ArduinoLink:
    def __init__(self, port="AUTO", baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False

        self.last_pwm = None
        self.last_dir = None
        self.last_send_time = 0.0

        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def _auto_find_port(self):
        ports = list(serial.tools.list_ports.comports())
        preferred_keywords = [
            "Arduino",
            "CH340",
            "CH341",
            "USB Serial",
            "USB2.0-Serial",
            "ttyACM",
            "ttyUSB",
        ]

        for p in ports:
            text = f"{p.device} {p.description} {p.manufacturer}"
            if any(k.lower() in text.lower() for k in preferred_keywords):
                return p.device

        if ports:
            return ports[0].device

        return None

    def connect(self):
        port = self.port
        if port == "AUTO":
            port = self._auto_find_port()

        if not port:
            print("[ARDUINO] No serial port found.", flush=True)
            return False

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=self.baud,
                timeout=ARDUINO_TIMEOUT_SEC,
                write_timeout=ARDUINO_TIMEOUT_SEC,
            )

            # Arduino resets when serial opens.
            time.sleep(2.0)

            self.connected = True
            self.running = True

            print(f"[ARDUINO] Connected on {port} at {self.baud} baud.", flush=True)

            self.thread = threading.Thread(target=self._read_worker, daemon=True)
            self.thread.start()

            self.stop()
            self.send_line("PING")
            self.send_line("E?")
            return True

        except Exception as e:
            self.ser = None
            self.connected = False
            print(f"[ARDUINO] Connect failed on {port}: {e}", flush=True)
            return False

    def send_line(self, line: str):
        if not self.connected or self.ser is None:
            return False

        try:
            with self.lock:
                self.ser.write((line.strip() + "\n").encode("ascii"))
            return True

        except serial.SerialTimeoutException:
            # Temporary timeout — port still alive, just retry next cycle
            print(f"[ARDUINO] Serial write timeout (skipping)", flush=True)
            return False
        except Exception as e:
            print(f"[ARDUINO] Serial write failed: {e}", flush=True)
            self.connected = False
            return False

    def set_motor(self, pwm: int, direction: int):
        pwm = int(np.clip(pwm, 0, 255))
        direction = 1 if int(direction) else 0

        # Avoid spamming identical serial commands every 20 ms.
        now = time.perf_counter()
        unchanged = (pwm == self.last_pwm and direction == self.last_dir)
        if unchanged and (now - self.last_send_time) < 0.10:
            return True

        ok = self.send_line(f"M {pwm} {direction}")

        if ok:
            self.last_pwm = pwm
            self.last_dir = direction
            self.last_send_time = now

        return ok

    def stop(self):
        return self.set_motor(0, 1)

    def _read_worker(self):
        while self.running:
            if not self.connected or self.ser is None:
                time.sleep(0.05)
                continue

            try:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("ascii", errors="ignore").strip()
                if line:
                    self._handle_line(line)

            except Exception as e:
                print(f"[ARDUINO] Read failed: {e}", flush=True)
                self.connected = False
                time.sleep(0.1)

    def _handle_line(self, line: str):
        global arduino_speed_mps, arduino_rpm, arduino_pulse_count
        global arduino_direction, pedal_percent, pedal_pressed, pedal_adc
        global pedal_voltage, manual_override, last_telemetry_time
        global _rpm_buf, _enc_rpm_smoothed, _rpm_last_enc_t

        parts = line.split()
        if not parts:
            return

        tag = parts[0]

        if tag == "T" and len(parts) >= 9:
            try:
                rpm = float(parts[1])
                speed = float(parts[2])
                pulses = int(float(parts[3]))
                direction = int(float(parts[4]))
                p_pct = float(parts[5])
                p_pressed = bool(int(float(parts[6])))
                p_adc = int(float(parts[7]))
                p_v = float(parts[8])

                # Smooth RPM over 8 samples to reject encoder noise bursts
                _rpm_buf.append(rpm)
                if len(_rpm_buf) > _RPM_SMOOTH_N:
                    _rpm_buf.pop(0)
                _enc_rpm_smoothed = sum(_rpm_buf) / len(_rpm_buf)
                arduino_rpm = _enc_rpm_smoothed
                _rpm_last_enc_t = time.perf_counter()

                arduino_speed_mps = speed
                arduino_pulse_count = pulses
                arduino_direction = direction
                pedal_percent = p_pct
                pedal_pressed = p_pressed
                pedal_adc = p_adc
                pedal_voltage = p_v
                manual_override = p_pressed
                last_telemetry_time = time.perf_counter()

            except ValueError:
                print(f"[ARDUINO] Bad telemetry: {line}", flush=True)

        elif tag == "P" and len(parts) >= 4:
            try:
                p_pressed = bool(int(float(parts[1])))
                p_pct = float(parts[2])
                p_adc = int(float(parts[3]))
                p_v = float(parts[4]) if len(parts) >= 5 else pedal_voltage

                pedal_pressed = p_pressed
                pedal_percent = p_pct
                pedal_adc = p_adc
                pedal_voltage = p_v
                manual_override = p_pressed

            except ValueError:
                print(f"[ARDUINO] Bad pedal line: {line}", flush=True)

        elif tag in ("READY", "PEDAL_CAL", "ERR", "SAFETY", "PEDAL"):
            print(f"[ARDUINO] {line}", flush=True)
        elif tag == "OK":
            # Suppress "OK M 0 1" motor-ack spam; only print non-motor OKs
            if len(parts) < 2 or parts[1] != "M":
                print(f"[ARDUINO] {line}", flush=True)

        else:
            # Keep this visible while testing.
            print(f"[ARDUINO] {line}", flush=True)

    def close(self):
        self.running = False
        try:
            self.stop()
            time.sleep(0.05)
        except Exception:
            pass

        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

        self.connected = False


_arduino = ArduinoLink(ARDUINO_PORT, ARDUINO_BAUD)


# ============================================================
# Drive motor functions — RPM closed-loop
# ============================================================
def set_drive_speed_mps(speed_mps: float):
    """Start the motor at the RPM that corresponds to speed_mps on a 20-inch wheel."""
    global _rpm_target, _rpm_pwm_actual, _rpm_boost_end
    if manual_override or speed_mps <= 0.0:
        stop_drive_motor()
        return
    target_rpm = (speed_mps / WHEEL_CIRCUMFERENCE_M) * 60.0
    if abs(target_rpm - _rpm_target) < 2.0:
        return  # target unchanged — don't retrigger boost
    _rpm_target = target_rpm
    _rpm_pwm_actual = _RPM_BOOST_PWM
    _rpm_boost_end = time.perf_counter() + _RPM_BOOST_DURATION_S


def stop_drive_motor():
    global _rpm_target, _rpm_pwm_actual, _rpm_boost_end
    _rpm_target = 0.0
    _rpm_pwm_actual = 0
    _rpm_boost_end = 0.0
    if USE_ARDUINO and _arduino.connected:
        _arduino.set_motor(0, 1)


def tick_rpm_controller():
    """One P-controller step. Safe to call from any thread — serial writes use _arduino.lock."""
    global _rpm_pwm_actual, _rpm_prev_enc_t
    if not USE_ARDUINO or not _arduino.connected:
        return
    if _rpm_target <= 0 or manual_override:
        _rpm_pwm_actual = 0
        _arduino.set_motor(0, 1)
        return
    now = time.perf_counter()
    # Exit boost early if speed already close to target (prevents overshoot jerk at boost end)
    if now < _rpm_boost_end and _enc_rpm_smoothed < _rpm_target * 0.85:
        _arduino.set_motor(_RPM_BOOST_PWM, 1)
        return
    # Only update PWM when a new telemetry packet has arrived (10 Hz from Arduino)
    if _rpm_last_enc_t > _rpm_prev_enc_t:
        _rpm_prev_enc_t = _rpm_last_enc_t
        error = _rpm_target - _enc_rpm_smoothed
        adj = int(_RPM_KP * error)
        _rpm_pwm_actual = int(max(_RPM_MIN_RUN_PWM, min(MAX_MOTOR_PWM, _rpm_pwm_actual + adj)))
    _arduino.set_motor(_rpm_pwm_actual, 1)


_motor_ctrl_stop = threading.Event()

def _motor_ctrl_loop():
    """Motor P-controller thread — runs at 10 Hz independent of YOLO frame rate."""
    while not _motor_ctrl_stop.is_set():
        t0 = time.perf_counter()
        tick_rpm_controller()
        elapsed = time.perf_counter() - t0
        _motor_ctrl_stop.wait(timeout=max(0.0, 0.10 - elapsed))  # target 10 Hz


# ============================================================
# GPIO callbacks
# ============================================================
def _pedal_gpio_callback(channel):
    global manual_override, pedal_pressed

    if not USE_PEDAL_GPIO_FROM_ARDUINO:
        return

    raw = GPIO.input(PIN_PEDAL_FROM_ARDUINO)

    if PEDAL_GPIO_ACTIVE_LOW:
        pressed = raw == GPIO.LOW
    else:
        pressed = raw == GPIO.HIGH

    pedal_pressed = pressed
    manual_override = pressed

    if pressed:
        stop_drive_motor()
        set_servo_deg(SERVO_AEB_RELEASE_DEG)
        print("[PEDAL GPIO] Manual override active", flush=True)
    else:
        print("[PEDAL GPIO] Manual override released", flush=True)


# ============================================================
# GPIO setup / teardown
# ============================================================
_servo_pwm = None
_last_servo_pulse_us = None


def _safe_add_event(pin, edge, callback, bouncetime=0):
    try:
        GPIO.remove_event_detect(pin)
    except Exception:
        pass

    try:
        if bouncetime:
            GPIO.add_event_detect(pin, edge, callback=callback, bouncetime=bouncetime)
        else:
            GPIO.add_event_detect(pin, edge, callback=callback)

    except RuntimeError as e:
        print(
            f"[GPIO] Warning: edge detection on pin {pin} failed ({e})",
            flush=True,
        )


def setup_gpio():
    global _servo_pwm

    if not _ON_RPI:
        print("[GPIO] Not running on Raspberry Pi. GPIO disabled.", flush=True)
        return

    GPIO.setwarnings(False)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)

    for pin in (PIN_BUZZER, PIN_LED, PIN_RELAY):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    for pin in (PIN_BTN1, PIN_BTN2, PIN_BTN3):
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    if USE_PEDAL_GPIO_FROM_ARDUINO:
        # Arduino D7 open-drain pedal output should connect here.
        # Pi uses pull-up to 3.3 V.
        pull = GPIO.PUD_UP if PEDAL_GPIO_ACTIVE_LOW else GPIO.PUD_DOWN
        GPIO.setup(PIN_PEDAL_FROM_ARDUINO, GPIO.IN, pull_up_down=pull)
        _safe_add_event(
            PIN_PEDAL_FROM_ARDUINO,
            GPIO.BOTH,
            _pedal_gpio_callback,
            bouncetime=50,
        )

    _release_pulse_us = int(SERVO_RELEASE_US + (SERVO_AEB_RELEASE_DEG / 100.0) * (SERVO_FULL_US - SERVO_RELEASE_US))
    if _USE_PIGPIO:
        _pigpio_pi.set_servo_pulsewidth(PIN_SERVO, _release_pulse_us)
        print("[GPIO] Servo on pigpio hardware timing.", flush=True)
    else:
        GPIO.setup(PIN_SERVO, GPIO.OUT)
        _servo_pwm = GPIO.PWM(PIN_SERVO, 50)
        _servo_pwm.start(_release_pulse_us / 20000.0 * 100.0)
        print("[GPIO] Servo on RPi.GPIO software PWM.", flush=True)


def cleanup_gpio():
    if _ON_RPI:
        try:
            set_warning_output(False)
            set_relay_output(False)
            set_servo_deg(SERVO_AEB_RELEASE_DEG)
            time.sleep(0.3)
        except Exception:
            pass

        try:
            if _USE_PIGPIO:
                _pigpio_pi.set_servo_pulsewidth(PIN_SERVO, 0)
                _pigpio_pi.stop()
            elif _servo_pwm:
                _servo_pwm.stop()
        except Exception:
            pass

        try:
            GPIO.cleanup()
        except Exception:
            pass

    _arduino.close()


# ============================================================
# Actuator outputs
# ============================================================
def set_relay_output(energize: bool):
    """
    energize=True  -> relay coil ON  -> power CUT
    energize=False -> relay coil OFF -> normal power ON

    WARNING:
    Only use this to cut the drive motor or unsafe actuator power.
    Do NOT cut servo-brake power during AEB braking if the servo needs power to hold brake.
    """
    if not _ON_RPI:
        return

    GPIO.output(PIN_RELAY, GPIO.HIGH if energize else GPIO.LOW)


def schedule_relay_cut(delay_s: float = 0.075):
    """
    Normal-stop sequence: stop motor first, then open relay after delay_s.
    Lets actively driven current decay before breaking the circuit.
    Emergency: call set_relay_output(True) directly — don't use this.
    """
    global _relay_open_after
    _relay_open_after = time.perf_counter() + delay_s


def cancel_relay_cut():
    global _relay_open_after
    _relay_open_after = None


def set_warning_output(enabled: bool):
    if not _ON_RPI:
        return

    state = GPIO.HIGH if enabled else GPIO.LOW
    GPIO.output(PIN_LED, state)
    GPIO.output(PIN_BUZZER, state)


def set_brake_output(level: float):
    """
    Servo brake only.
    This does NOT control the MDD10 drive motor.
    """
    level = float(np.clip(level, 0.0, 1.0))
    set_servo_position(level)


def set_servo_position(level: float):
    global _last_servo_pulse_us, _pending_latency_t, _latency_log
    level = float(np.clip(level, 0.0, 1.0))
    pulse_us = int(SERVO_RELEASE_US + level * (SERVO_FULL_US - SERVO_RELEASE_US))

    prev_pulse = _last_servo_pulse_us
    if pulse_us == prev_pulse:
        _pending_latency_t = None
        return
    _last_servo_pulse_us = pulse_us

    if not _ON_RPI:
        _pending_latency_t = None
        return

    if _USE_PIGPIO and _pigpio_pi is not None:
        _pigpio_pi.set_servo_pulsewidth(PIN_SERVO, pulse_us)
    elif _servo_pwm is not None:
        _servo_pwm.ChangeDutyCycle(pulse_us / 20000.0 * 100.0)

    if _pending_latency_t is not None and prev_pulse is not None:
        t_cmd = time.perf_counter()
        tau_elec = t_cmd - _pending_latency_t
        delta_pulse = abs(pulse_us - prev_pulse)
        phys_deg = delta_pulse / SERVO_PHYS_PULSE_RANGE_US * SERVO_PHYS_ANGLE_DEG
        tau_mech = phys_deg / SERVO_SPEED_DEG_PER_S
        tau_sys  = tau_elec + tau_mech
        record = {
            "t_trigger": _pending_latency_t,
            "tau_electronic_ms": round(tau_elec * 1000, 3),
            "tau_mechanical_ms": round(tau_mech * 1000, 3),
            "tau_system_ms":     round(tau_sys  * 1000, 3),
            "delta_pulse_us":    delta_pulse,
            "phys_deg":          round(phys_deg, 2),
        }
        _latency_log.append(record)
        print(
            f"[LATENCY] τ_elec={tau_elec*1000:.2f}ms  "
            f"τ_mech={tau_mech*1000:.1f}ms  "
            f"τ_sys={tau_sys*1000:.1f}ms  "
            f"(Δ{delta_pulse}µs = {phys_deg:.1f}° physical)",
            flush=True,
        )
        _pending_latency_t = None


def set_servo_deg(deg: float):
    """Move servo to a specific angle in degrees (0°=release, 85°=full AEB, 70°=partial AEB)."""
    set_servo_position(float(np.clip(deg, 0.0, 100.0)) / 100.0)


# ============================================================
# Vision thread
# ============================================================
class _VisionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.distance = None
        self.frame = None
        self.running = True


_vis = _VisionState()


def _vision_worker():
    frame_count = 0
    t_start = time.perf_counter()

    for frame, tracks in vision_main(yield_every_frame=True):
        if not _vis.running:
            break

        frame_count += 1
        elapsed = time.perf_counter() - t_start
        if elapsed >= 5.0:
            print(
                f"[YOLO] {frame_count / elapsed:.1f} FPS "
                f"({1000 * elapsed / frame_count:.1f} ms/frame)",
                flush=True,
            )
            frame_count = 0
            t_start = time.perf_counter()

        dist = _get_closest_live_distance(tracks)

        with _vis.lock:
            _vis.distance = dist
            _vis.frame = frame


# ============================================================
# Helpers
# ============================================================
def depth_to_meters(depth_value: float) -> float:
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


# ============================================================
# Dashboard display helpers
# ============================================================
def _resize_letterbox(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - nw) // 2
    pad_y = (target_h - nh) // 2
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out


def _status_color(state_str):
    return {
        "IDLE":      (160, 160, 160),
        "INIT":      (200, 200,   0),
        "RUN":       (  0, 220,   0),
        "FCW":       (  0, 200, 255),
        "PARTIAL":   (  0, 165, 255),
        "EMERGENCY": (  0,   0, 255),
        "STOP":      (  0, 255, 255),
        "CRASH":     (  0,   0, 255),
    }.get(state_str, (200, 200, 200))


def _draw_label_value(img, y, label, value,
                      lcolor=(180, 180, 180), vcolor=(255, 255, 255), scale=0.75):
    cv2.putText(img, label, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, lcolor, 1, cv2.LINE_AA)
    cv2.putText(img, str(value), (230, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, vcolor, 2, cv2.LINE_AA)


def _draw_status_panel(panel):
    state_str   = _ds.get("state", "IDLE")
    curr_speed  = _ds.get("current_speed", 0.0)
    vdist       = _ds.get("virtual_distance", None)
    _ttc        = _ds.get("ttc", math.inf)
    _brake_on   = _ds.get("brake_on", False)
    _warning_on = _ds.get("warning_on", False)
    _brake_lvl  = _ds.get("brake_level", 0.0)
    _manual_ov  = _ds.get("manual_override", False)
    _tgt_speed  = _ds.get("manual_speed_mps", 0.0)
    _pedal_pct  = _ds.get("pedal_percent", 0.0)

    h, w = panel.shape[:2]
    col = _status_color(state_str)

    cv2.rectangle(panel, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(panel, "AEB DEMO DASHBOARD", (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 70), (w, 155), col, -1)
    cv2.putText(panel, state_str, (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 5, cv2.LINE_AA)

    ttc_str  = f"{_ttc:.2f} s" if math.isfinite(_ttc) else "---"
    dist_str = f"{vdist:.2f} m" if vdist is not None else "---"
    spd_str  = f"{curr_speed:.2f} m/s  ({curr_speed * 3.6:.1f} km/h)"
    tgt_str  = f"{_tgt_speed:.2f} m/s  ({_tgt_speed * 3.6:.1f} km/h)"

    y, dy = 200, 52
    for lbl, val, vc in [
        ("Speed:",    spd_str,                        col),
        ("Target:",   tgt_str,                        (200, 200, 200)),
        ("Distance:", dist_str,                       (255, 255, 255)),
        ("TTC:",      ttc_str,                        (0, 255, 255) if math.isfinite(_ttc) else (160, 160, 160)),
        ("Brake:",    "ON" if _brake_on else "off",   (0, 0, 255) if _brake_on else (160, 160, 160)),
        ("Warning:",  "ON" if _warning_on else "off", (0, 200, 255) if _warning_on else (160, 160, 160)),
        ("Override:", "YES" if _manual_ov else "no",  (0, 0, 255) if _manual_ov else (160, 160, 160)),
        ("Pedal:",    f"{_pedal_pct:.1f}%",           (200, 200, 200)),
    ]:
        _draw_label_value(panel, y, lbl, val, vcolor=vc)
        y += dy

    bar_y = h - 80
    bar_w = int((w - 40) * _brake_lvl)
    cv2.rectangle(panel, (20, bar_y), (w - 20, bar_y + 30), (50, 50, 50), -1)
    if bar_w > 0:
        bar_col = (0, 0, 255) if _brake_lvl > 0.6 else (0, 165, 255)
        cv2.rectangle(panel, (20, bar_y), (20 + bar_w, bar_y + 30), bar_col, -1)
    cv2.putText(panel, f"Brake: {int(_brake_lvl * 100)}%",
                (20, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_live_graph(panel):
    _time_log  = _ds.get("time_log", [])
    _speed_log = _ds.get("speed_log", [])

    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w, h), (15, 15, 15), -1)
    cv2.putText(panel, "Speed vs Time", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)

    if len(_time_log) < 2:
        cv2.putText(panel, "Waiting for data...", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
        return

    t_now = _time_log[-1]
    t_min = max(t_now - GRAPH_HISTORY_SEC, _time_log[0])
    t_max = t_now

    idx_start = 0
    for i, t in enumerate(_time_log):
        if t >= t_min:
            idx_start = i
            break

    t_win = _time_log[idx_start:]
    s_win = _speed_log[idx_start:]
    if not t_win:
        return

    s_max = max(max(s_win), 1.0)
    ml, mr, mt, mb = 60, 20, 50, 40
    pw = w - ml - mr
    ph = h - mt - mb

    def px(t_val):
        return ml + int((t_val - t_min) / max(t_max - t_min, 1e-6) * pw)

    def py(s_val):
        return mt + ph - int(s_val / s_max * ph)

    for i in range(5):
        cv2.line(panel, (ml + int(i / 4 * pw), mt), (ml + int(i / 4 * pw), mt + ph), (40, 40, 40), 1)
    for i in range(5):
        cv2.line(panel, (ml, mt + int(i / 4 * ph)), (ml + pw, mt + int(i / 4 * ph)), (40, 40, 40), 1)

    cv2.line(panel, (ml, mt), (ml, mt + ph), (100, 100, 100), 2)
    cv2.line(panel, (ml, mt + ph), (ml + pw, mt + ph), (100, 100, 100), 2)

    pts = [(px(t), py(s)) for t, s in zip(t_win, s_win)]
    for i in range(1, len(pts)):
        cv2.line(panel, pts[i - 1], pts[i], (0, 220, 80), 2, cv2.LINE_AA)

    cv2.putText(panel, f"{s_max:.1f}", (5, mt + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "0.0", (5, mt + ph),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


def _draw_ego_animation(panel):
    _state   = _ds.get("state", "IDLE")
    _lock_d  = _ds.get("locked_initial_distance", None)
    _robot_z = _ds.get("robot_z", 0.0)

    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w, h), (10, 10, 30), -1)

    road_y1 = int(h * 0.38)
    road_y2 = int(h * 0.62)
    cv2.rectangle(panel, (0, road_y1), (w, road_y2), (50, 50, 50), -1)
    cy = (road_y1 + road_y2) // 2

    for i in range(0, w, 60):
        cv2.rectangle(panel, (i, cy - 3), (i + 30, cy + 3), (180, 180, 0), -1)

    span_m = max(_lock_d if _lock_d else 5.0, 5.0)
    margin = 80
    road_w = w - 2 * margin

    rx = margin + int((_robot_z / span_m) * road_w)
    rx = min(max(rx, margin), w - margin - BOX_W)

    obj_x = margin + road_w
    if _lock_d is not None:
        obj_x = margin + int((_lock_d / span_m) * road_w)
        obj_x = min(max(obj_x, margin), w - margin - BOX_W)

    robot_col = (0, 200, 0) if _state != "CRASH" else (0, 0, 255)
    cv2.rectangle(panel, (rx, road_y1 + 5), (rx + BOX_W, road_y2 - 5), robot_col, -1)
    cv2.putText(panel, "EGO", (rx + 4, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    if _lock_d is not None:
        cv2.rectangle(panel, (obj_x, road_y1 + 5), (obj_x + BOX_W, road_y2 - 5), (0, 0, 200), -1)
        cv2.putText(panel, "OBJ", (obj_x + 4, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        gap_col = (
            (0, 255, 0) if _state in ("RUN", "IDLE", "INIT")
            else (0, 165, 255) if _state == "FCW"
            else (0, 0, 255)
        )
        cv2.line(panel, (rx + BOX_W, cy), (obj_x, cy), gap_col, 3, cv2.LINE_AA)

    col = _status_color(_state)
    cv2.putText(panel, _state, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3, cv2.LINE_AA)


def _exit_button_rect():
    x1 = SCREEN_W - EXIT_BUTTON_W - EXIT_BUTTON_MARGIN
    y1 = EXIT_BUTTON_MARGIN
    x2 = SCREEN_W - EXIT_BUTTON_MARGIN
    y2 = EXIT_BUTTON_MARGIN + EXIT_BUTTON_H
    return x1, y1, x2, y2


def _draw_exit_button(img):
    x1, y1, x2, y2 = _exit_button_rect()
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 2)
    cv2.putText(img, "EXIT", (x1 + 20, y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def _mouse_callback(event, x, y, flags, param):
    global _exit_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = _exit_button_rect()
        if x1 <= x <= x2 and y1 <= y <= y2:
            _exit_requested = True


def _make_dashboard(cam_frame):
    dash = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    cam_panel = _resize_letterbox(cam_frame, CAM_W, TOP_H)
    dash[0:TOP_H, 0:CAM_W] = cam_panel

    status_panel = np.zeros((TOP_H, STATUS_W, 3), dtype=np.uint8)
    _draw_status_panel(status_panel)
    dash[0:TOP_H, CAM_W:CAM_W + STATUS_W] = status_panel

    graph_panel = np.zeros((BOTTOM_H, GRAPH_W, 3), dtype=np.uint8)
    _draw_live_graph(graph_panel)
    dash[TOP_H:SCREEN_H, 0:GRAPH_W] = graph_panel

    ego_panel = np.zeros((BOTTOM_H, EGO_W, 3), dtype=np.uint8)
    _draw_ego_animation(ego_panel)
    dash[TOP_H:SCREEN_H, GRAPH_W:SCREEN_W] = ego_panel

    _draw_exit_button(dash)
    return dash


def _show(cam_frame, world_img):
    if DASHBOARD_MODE:
        dash = _make_dashboard(cam_frame)
        cv2.imshow("AEB System", dash)
    elif TV_MODE:
        cam_h, cam_w = cam_frame.shape[:2]
        w_h = cam_w * WORLD_HEIGHT // WORLD_WIDTH
        world_scaled = cv2.resize(world_img, (cam_w, w_h))
        combined = np.vstack([cam_frame, world_scaled])
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


def speed_to_motor_pwm(speed_mps: float) -> int:
    """Fixed PWM lookup from calibrated presets. Encoder is read but not used for control."""
    for i, s in enumerate(SPEED_PRESETS_MPS):
        if abs(speed_mps - s) < 0.05:
            return SPEED_PRESETS_PWM[i]
    return 0


def read_wheel_speed_mps() -> float:
    """
    Correct speed source:
    - Arduino encoder telemetry.
    - If telemetry is stale, return 0 for safety.
    - If Arduino is not connected and debugging is allowed, use manual speed fallback.
    """
    now = time.perf_counter()

    if USE_ARDUINO and _arduino.connected:
        if last_telemetry_time > 0 and (now - last_telemetry_time) <= ARDUINO_TELEMETRY_STALE_SEC:
            return max(float(arduino_speed_mps), 0.0)
        return 0.0

    if ALLOW_RUN_WITHOUT_ARDUINO:
        return max(manual_speed_mps, 0.0)

    return 0.0


def refresh_manual_override_from_telemetry():
    """
    Main manual override source is Arduino analog pedal telemetry.
    Optional backup source is Arduino D7 open-drain output to Pi GPIO.
    """
    global manual_override

    manual_override = bool(pedal_pressed)

    if USE_PEDAL_GPIO_FROM_ARDUINO and _ON_RPI:
        raw = GPIO.input(PIN_PEDAL_FROM_ARDUINO)
        gpio_pressed = (raw == GPIO.LOW) if PEDAL_GPIO_ACTIVE_LOW else (raw == GPIO.HIGH)
        manual_override = manual_override or gpio_pressed


def command_drive_for_state(state: str, requested_speed_mps: float):
    """
    Physical motor policy:
    - RUN / FCW: drive motor according to requested demo speed.
    - PARTIAL / EMERGENCY / STOP / CRASH / manual override: stop drive motor.
    """
    if manual_override:
        stop_drive_motor()
        return

    if state in ("RUN", "FCW"):
        set_drive_speed_mps(requested_speed_mps)
    else:
        stop_drive_motor()


# ============================================================
# Plot
# ============================================================
def plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log):
    if not time_log:
        return

    fcw_idx = first_state_idx(state_log, "FCW")
    partial_idx = first_state_idx(state_log, "PARTIAL")
    emergency_idx = first_state_idx(state_log, "EMERGENCY")
    stop_idx = first_state_idx(state_log, "STOP")
    crash_idx = first_state_idx(state_log, "CRASH")

    markers = [
        (fcw_idx, "FCW trigger"),
        (partial_idx, "Partial brake"),
        (emergency_idx, "Emergency brake"),
        (stop_idx, "Stop"),
        (crash_idx, "Crash"),
    ]

    def _vlines():
        for idx, lbl in markers:
            if idx is not None:
                plt.axvline(time_log[idx], linestyle="--", label=lbl)

    plt.figure(figsize=(9, 4))
    plt.plot(time_log, distance_log, label="Remaining distance")
    _vlines()
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distance vs Time")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(9, 4))
    plt.plot(time_log, speed_log, label="Encoder speed")
    _vlines()
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed vs Time")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(9, 4))
    ttc_arr = np.array(ttc_log, dtype=float)
    finite_ttc = ttc_arr[np.isfinite(ttc_arr)]
    y_top = max(float(np.max(finite_ttc)), SAFE_TTC + 0.5) if finite_ttc.size > 0 else SAFE_TTC + 1.0

    plt.axhspan(0, PARTIAL_TTC, color="red", alpha=0.20, label="Emergency")
    plt.axhspan(PARTIAL_TTC, FCW_TTC, color="orange", alpha=0.20, label="Partial")
    plt.axhspan(FCW_TTC, SAFE_TTC, color="yellow", alpha=0.20, label="FCW")
    plt.axhspan(SAFE_TTC, y_top, color="green", alpha=0.12, label="Safe")

    for thresh in (SAFE_TTC, FCW_TTC, PARTIAL_TTC):
        plt.axhline(thresh, linestyle="--", color="blue")

    plt.plot(time_log, ttc_log, color="black", label="TTC")
    _vlines()
    plt.xlabel("Time (s)")
    plt.ylabel("TTC (s)")
    plt.title("TTC vs Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    import datetime, os, csv
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(__file__)

    out_path = os.path.join(out_dir, f"aeb_plot_{ts}.png")
    plt.savefig(out_path, dpi=100)
    plt.close("all")
    print(f"[PLOT] Saved to {out_path}", flush=True)

    if _latency_log:
        csv_path = os.path.join(out_dir, f"latency_{ts}.csv")
        fields = ["t_trigger", "tau_electronic_ms", "tau_mechanical_ms", "tau_system_ms",
                  "delta_pulse_us", "phys_deg"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(_latency_log)
        print(f"[LATENCY] CSV saved to {csv_path}", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    global manual_speed_mps, _speed_preset_idx, manual_override
    global pedal_pressed, pedal_percent, pedal_adc, pedal_voltage
    global _pending_latency_t, _latency_log

    ego = EgoVehicle()
    ego.set_speed(0.0)

    state = "IDLE"

    locked_initial_distance = None
    virtual_distance = None
    robot_z = 0.0

    init_samples = []
    init_start_time = None
    _init_dist_input = ""

    brake_on = False
    warning_on = False
    brake_level = 0.0
    plots_shown = False

    brake_trigger_speed = 0.0
    brake_required_stop_distance = 0.0
    brake_trigger_mode = None

    last_live_distance = None
    _btn_press_time = [0.0, 0.0, 0.0]  # debounce timestamps per button
    _prev_manual_override = False

    time_log = []
    distance_log = []
    speed_log = []
    ttc_log = []
    travel_log = []
    stop_req_log = []
    state_log = []
    sim_time = 0.0

    current_speed = 0.0
    ttc = math.inf
    status = "SAFE"

    setup_gpio()

    if USE_ARDUINO:
        connected = _arduino.connect()
        if not connected and not ALLOW_RUN_WITHOUT_ARDUINO:
            raise RuntimeError("Arduino controller is not connected.")

    _motor_ctrl_stop.clear()
    _motor_ctrl_thread = threading.Thread(target=_motor_ctrl_loop, daemon=True, name="motor-ctrl")
    _motor_ctrl_thread.start()
    print("[MOTOR] Control thread started at 10 Hz (independent of YOLO).", flush=True)

    if TV_MODE or DASHBOARD_MODE:
        cv2.namedWindow("AEB System", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("AEB System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if DASHBOARD_MODE:
            cv2.setMouseCallback("AEB System", _mouse_callback)

    vision_thread = threading.Thread(target=_vision_worker, daemon=True)
    vision_thread.start()

    print("Waiting for camera...", flush=True)
    while True:
        with _vis.lock:
            ready = _vis.frame is not None
        if ready:
            break
        time.sleep(0.05)

    print("Camera ready. Starting control loop at 50 Hz.", flush=True)

    loop_counter = 0

    try:
        while True:
            loop_t0 = time.perf_counter()
            loop_counter += 1
            should_display = (loop_counter % DISPLAY_EVERY_N == 0)
            timings = {}

            # ------------------------------------------------
            # Refresh Arduino/pedal state
            # ------------------------------------------------
            refresh_manual_override_from_telemetry()

            brake_level = 0.0

            # Braking debug: print on change or every ~1 s (50 loops)
            _brake_dbg_changed = (manual_override != _prev_manual_override)
            if _brake_dbg_changed or (loop_counter % 50 == 0):
                servo_us = _last_servo_pulse_us if _last_servo_pulse_us is not None else SERVO_RELEASE_US
                print(
                    f"[BRAKE] pedal={pedal_percent:5.1f}%  adc={pedal_adc:4d}  V={pedal_voltage:.2f}"
                    f"  override={'YES' if manual_override else 'no ':3s}"
                    f"  servo={servo_us}us  brake={brake_level*100:.0f}%"
                    f"  state={state}",
                    flush=True,
                )

            # Motor override: stop motor on rising edge, restore on falling edge
            if manual_override and not _prev_manual_override:
                stop_drive_motor()
                schedule_relay_cut(0.075)

            if not manual_override and _prev_manual_override:
                cancel_relay_cut()
                set_relay_output(False)
                manual_speed_mps = 0.0
                stop_drive_motor()

            _prev_manual_override = manual_override

            # Fire scheduled relay cut if timer has expired
            if _relay_open_after is not None and time.perf_counter() >= _relay_open_after:
                cancel_relay_cut()
                set_relay_output(True)

            # ------------------------------------------------
            # Snapshot latest vision data
            # ------------------------------------------------
            t0 = time.perf_counter()
            with _vis.lock:
                live_distance = _vis.distance
                cam_frame = _vis.frame
            last_live_distance = live_distance if live_distance is not None else last_live_distance
            timings["vision"] = time.perf_counter() - t0

            # ------------------------------------------------
            # Keyboard + physical buttons
            # ------------------------------------------------
            t0 = time.perf_counter()

            # Poll buttons BEFORE cv2.waitKey — waitKey on Pi blocks for
            # 100-400ms (slow OpenCV display), so any tap shorter than that
            # would be invisible if polled after.
            _btn2_key = _btn3_key = False
            if _ON_RPI:
                b1 = GPIO.input(PIN_BTN1) == GPIO.LOW
                b2 = GPIO.input(PIN_BTN2) == GPIO.LOW
                b3 = GPIO.input(PIN_BTN3) == GPIO.LOW
                now_t = time.perf_counter()
                debounce = BTN_BOUNCE_MS / 1000.0

                if b1 and not _btn_state[0] and (now_t - _btn_press_time[0]) > debounce:
                    _btn_press_time[0] = now_t
                    was_stopped = (manual_speed_mps == 0.0)
                    _speed_preset_idx = (_speed_preset_idx + 1) % len(SPEED_PRESETS_MPS)
                    manual_speed_mps = SPEED_PRESETS_MPS[_speed_preset_idx]
                    print(f"[BTN1] Target speed -> {manual_speed_mps:.2f} m/s "
                          f"({manual_speed_mps * 3.6:.0f} km/h)", flush=True)
                    if was_stopped:
                        # Motor starting — reset graph so recording begins from this moment
                        time_log.clear(); distance_log.clear(); speed_log.clear()
                        ttc_log.clear(); travel_log.clear(); stop_req_log.clear(); state_log.clear()
                        sim_time = 0.0
                        _latency_log.clear()

                if b2 and not _btn_state[1] and (now_t - _btn_press_time[1]) > debounce:
                    _btn_press_time[1] = now_t
                    _btn2_key = True
                    print(f"[BTN2] pressed  state={state}", flush=True)

                if b3 and not _btn_state[2] and (now_t - _btn_press_time[2]) > debounce:
                    _btn_press_time[2] = now_t
                    _btn3_key = True
                    print(f"[BTN3] pressed  state={state}", flush=True)

                _btn_state[0], _btn_state[1], _btn_state[2] = b1, b2, b3

            key = (cv2.waitKey(1) & 0xFF) if should_display else 0xFF

            if _btn2_key:
                key = ord("i")
            if _btn3_key:
                key = ord("r")

            if key == ord("q") or _exit_requested:
                break

            elif key == ord("i") and state in ("IDLE", "STOP", "CRASH"):
                prev_state = state
                state = "INIT"
                init_samples = []
                init_start_time = time.perf_counter()

                locked_initial_distance = None
                virtual_distance = None
                robot_z = 0.0

                brake_on = False
                warning_on = False
                brake_level = 0.0
                plots_shown = False

                brake_trigger_speed = 0.0
                brake_required_stop_distance = 0.0
                brake_trigger_mode = None

                # Only reset graph when coming from STOP/CRASH (new attempt).
                # Coming from IDLE keeps the graph continuous from motor start.
                if prev_state in ("STOP", "CRASH"):
                    time_log.clear(); distance_log.clear(); speed_log.clear()
                    ttc_log.clear(); travel_log.clear(); stop_req_log.clear(); state_log.clear()
                    sim_time = 0.0

                _init_dist_input = ""

                set_warning_output(False)
                set_servo_deg(SERVO_AEB_RELEASE_DEG)
                set_relay_output(False)
                # Motor keeps running at preview speed — no stop_drive_motor() here
                reset_ego(ego)

            elif key == ord("i") and state == "INIT":
                # Confirm distance and enter RUN.
                #
                # Important behavior:
                # - If distance is already available from manual input or INIT camera samples,
                #   lock it now and start the simulation normally.
                # - If distance is NOT available, still enter RUN and keep the motor running,
                #   but do NOT start TTC/simulation until the camera detects a valid distance.
                dist_to_use = None

                if _init_dist_input:
                    try:
                        v = float(_init_dist_input)
                        if v > 0:
                            dist_to_use = v
                    except ValueError:
                        pass

                if dist_to_use is None and init_samples:
                    dist_to_use = float(np.mean(init_samples))

                locked_initial_distance = dist_to_use
                virtual_distance = locked_initial_distance if locked_initial_distance is not None else None
                robot_z = 0.0

                brake_on = False
                warning_on = False
                brake_level = 0.0
                ttc = math.inf
                status = "WAIT_DISTANCE" if locked_initial_distance is None else "SAFE"

                set_warning_output(False)
                set_servo_deg(SERVO_AEB_RELEASE_DEG)
                set_relay_output(False)
                reset_ego(ego)

                # Keep the physical motor running at the selected speed.
                # The simulation/TTC may still be waiting for a valid distance.
                command_drive_for_state("RUN", manual_speed_mps)

                if locked_initial_distance is not None:
                    print(
                        f"[INIT] Distance locked: {locked_initial_distance:.2f} m "
                        f"({'manual' if _init_dist_input else 'camera'})",
                        flush=True,
                    )
                else:
                    print(
                        "[INIT] No distance yet — entering RUN, motor running, "
                        "simulation waiting for first camera detection.",
                        flush=True,
                    )

                state = "RUN"

            elif state == "INIT" and key in [ord(c) for c in "0123456789."]:
                _init_dist_input += chr(key)

            elif state == "INIT" and key == 8:  # backspace
                _init_dist_input = _init_dist_input[:-1]

            elif key == ord("r"):
                state = "IDLE"
                init_samples = []
                init_start_time = None

                locked_initial_distance = None
                virtual_distance = None
                robot_z = 0.0

                brake_on = False
                warning_on = False
                brake_level = 0.0
                plots_shown = False

                brake_trigger_speed = 0.0
                brake_required_stop_distance = 0.0
                brake_trigger_mode = None

                time_log.clear()
                distance_log.clear()
                speed_log.clear()
                ttc_log.clear()
                travel_log.clear()
                stop_req_log.clear()
                state_log.clear()

                sim_time = 0.0
                current_speed = 0.0

                manual_speed_mps = 0.0
                _speed_preset_idx = len(SPEED_PRESETS_MPS) - 1

                cancel_relay_cut()
                set_warning_output(False)
                set_servo_deg(SERVO_AEB_RELEASE_DEG)
                set_relay_output(False)
                stop_drive_motor()
                reset_ego(ego)

            else:
                if key == ord("w"):
                    manual_speed_mps = min(manual_speed_mps + MANUAL_SPEED_STEP, MAX_DEMO_SPEED)
                elif key == ord("s"):
                    manual_speed_mps = max(manual_speed_mps - MANUAL_SPEED_STEP, 0.0)
                elif key == ord(" "):
                    manual_speed_mps = 0.0
                    stop_drive_motor()

            timings["key"] = time.perf_counter() - t0

            # ------------------------------------------------
            # IDLE
            # ------------------------------------------------
            if state == "IDLE":
                # Motor preview: run at selected speed so user can verify before starting AEB
                if manual_speed_mps > 0:
                    set_drive_speed_mps(manual_speed_mps)
                    current_speed = max(0.0, read_wheel_speed_mps())
                    virtual_distance = live_distance if live_distance is not None else 0.0
                    ttc = float("inf")
                else:
                    stop_drive_motor()

                t0 = time.perf_counter()
                world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

                cv2.putText(world, "IDLE - press I / Button2 to initialize",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(world, f"Target speed: {manual_speed_mps:.2f} m/s ({manual_speed_mps*3.6:.0f} km/h)",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                cv2.putText(world, f"Encoder speed: {arduino_speed_mps:.2f} m/s  RPM: {arduino_rpm:.1f}",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 2)

                dist_str = f"{last_live_distance:.2f} m" if last_live_distance else "none"
                cv2.putText(world, f"Live camera distance: {dist_str}",
                            (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 2)

                ard = "connected" if _arduino.connected else "not connected"
                cv2.putText(world, f"Arduino: {ard}  Pedal: {pedal_percent:.1f}%  override={int(manual_override)}",
                            (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

                cv2.putText(world, "W/S target speed, SPACE stop, Q quit",
                            (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                if should_display and cam_frame is not None:
                    _ds.update({
                        "state": state, "current_speed": current_speed,
                        "virtual_distance": virtual_distance, "ttc": ttc,
                        "brake_on": brake_on, "warning_on": warning_on,
                        "brake_level": brake_level, "manual_override": manual_override,
                        "manual_speed_mps": manual_speed_mps, "pedal_percent": pedal_percent,
                        "time_log": time_log, "speed_log": speed_log,
                        "locked_initial_distance": locked_initial_distance, "robot_z": robot_z,
                    })
                    _show(cam_frame, world)

                timings["render"] = time.perf_counter() - t0

                if loop_counter % PRINT_EVERY_N_LOOPS == 0:
                    print_timing("IDLE", timings)

                time.sleep(max(0.0, DT - (time.perf_counter() - loop_t0)))
                if manual_speed_mps == 0.0:
                    continue  # motor stopped — nothing to log yet

            # ------------------------------------------------
            # INIT
            # ------------------------------------------------
            if state == "INIT":
                # Motor keeps running at preview speed during distance locking —
                # no restart jolt when transitioning to RUN.
                set_drive_speed_mps(manual_speed_mps)

                t0 = time.perf_counter()

                if live_distance is not None:
                    init_samples.append(live_distance)

                elapsed = (time.perf_counter() - init_start_time) if init_start_time else 0.0
                world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

                cv2.putText(world, "INIT - locking initial distance",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(world, f"Samples: {len(init_samples)}/{INIT_REQUIRED_SAMPLES}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(world, f"Elapsed: {elapsed:.2f} s",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

                dist_str = f"{live_distance:.2f} m" if live_distance else "no detection"
                cam_avg  = f"{np.mean(init_samples):.2f} m" if init_samples else "---"
                cv2.putText(world, f"Camera: {dist_str}  avg({len(init_samples)}): {cam_avg}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 2)

                input_display = _init_dist_input if _init_dist_input else "(use camera avg)"
                cv2.putText(world, f"Manual dist: {input_display} m",
                            (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 255) if _init_dist_input else (120, 120, 120), 2)
                cv2.putText(world, "Type distance + press BTN2/I to start",
                            (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                if should_display and cam_frame is not None:
                    _ds.update({
                        "state": state, "current_speed": current_speed,
                        "virtual_distance": virtual_distance, "ttc": ttc,
                        "brake_on": brake_on, "warning_on": warning_on,
                        "brake_level": brake_level, "manual_override": manual_override,
                        "manual_speed_mps": manual_speed_mps, "pedal_percent": pedal_percent,
                        "time_log": time_log, "speed_log": speed_log,
                        "locked_initial_distance": locked_initial_distance, "robot_z": robot_z,
                    })
                    _show(cam_frame, world)

                timings["render"] = time.perf_counter() - t0

                if loop_counter % PRINT_EVERY_N_LOOPS == 0:
                    print_timing("INIT", timings)

                time.sleep(max(0.0, DT - (time.perf_counter() - loop_t0)))
                # No continue — fall through to logging so INIT appears on the graph

            # ------------------------------------------------
            # RUN / FCW / PARTIAL / EMERGENCY / STOP / CRASH
            # ------------------------------------------------
            t0 = time.perf_counter()
            wheel_speed_mps = read_wheel_speed_mps()
            timings["read_speed"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            virtual_distance = (
                max(locked_initial_distance - robot_z, 0.0)
                if locked_initial_distance is not None else None
            )
            timings["dist_calc"] = time.perf_counter() - t0

            t0 = time.perf_counter()

            if state == "INIT":
                # Still locking distance — record encoder speed and live distance so
                # the graph shows the approach phase before RUN begins.
                current_speed = max(0.0, wheel_speed_mps)
                virtual_distance = live_distance if live_distance is not None else 0.0
                ttc = float("inf")

            elif state in ("RUN", "FCW"):
                if not manual_override:
                    cancel_relay_cut()
                    set_relay_output(False)   # ensure relay closed while motor should run

                # Clamp to physical max to prevent EMI encoder spikes from collapsing virtual_distance.
                current_speed = float(np.clip(wheel_speed_mps, 0.0, MAX_DEMO_SPEED + 1.0))
                ego.set_speed(current_speed)

                # Always keep the physical motor running in RUN/FCW.
                # This is independent from whether the simulation distance is armed yet.
                command_drive_for_state("RUN", manual_speed_mps)

                if locked_initial_distance is None:
                    # RUN is active and the motor keeps running, but the simulation/TTC
                    # is not allowed to start until a valid camera distance appears.
                    state = "RUN"
                    warning_on = False
                    brake_on = False
                    brake_level = 0.0
                    ttc = math.inf
                    status = "WAIT_DISTANCE"
                    robot_z = 0.0
                    virtual_distance = live_distance if live_distance is not None else None

                    set_warning_output(False)
                    set_servo_deg(SERVO_AEB_RELEASE_DEG)

                    if live_distance is not None:
                        locked_initial_distance = float(live_distance)
                        virtual_distance = locked_initial_distance
                        robot_z = 0.0
                        reset_ego(ego)
                        ttc = math.inf
                        status = "SAFE"

                        print(
                            f"[RUN] First valid distance detected: "
                            f"{locked_initial_distance:.2f} m. Simulation starts now.",
                            flush=True,
                        )

                    # Do not calculate TTC or advance robot_z in this same loop.
                    # TTC starts on the next loop after distance is locked.

                else:
                    virtual_distance = max(locked_initial_distance - robot_z, 0.0)
                    ttc = ttc_from(virtual_distance, current_speed)
                    status = ttc_status(ttc)

                    if status == "SAFE":
                        state = "RUN"
                        warning_on = False
                        brake_on = False

                        set_warning_output(False)
                        command_drive_for_state(state, manual_speed_mps)

                        robot_z += current_speed * DT

                    elif status == "FCW":
                        state = "FCW"
                        warning_on = True
                        brake_on = False

                        set_warning_output(True)
                        command_drive_for_state(state, manual_speed_mps)

                        robot_z += current_speed * DT

                    elif status == "PARTIAL":
                        state = "PARTIAL"
                        warning_on = True
                        brake_on = True
                        brake_level = (SERVO_AEB_PARTIAL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)

                        brake_trigger_speed = current_speed
                        brake_trigger_mode = "PARTIAL"
                        brake_required_stop_distance = stopping_distance(current_speed, PARTIAL_BRAKE_DECEL)

                        set_warning_output(True)
                        stop_drive_motor()
                        schedule_relay_cut(0.075)
                        _pending_latency_t = time.perf_counter()   # τ_trigger for latency
                        set_servo_deg(SERVO_AEB_PARTIAL_DEG)

                    else:
                        state = "EMERGENCY"
                        warning_on = True
                        brake_on = True
                        brake_level = (SERVO_AEB_FULL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)

                        brake_trigger_speed = current_speed
                        brake_trigger_mode = "EMERGENCY"
                        brake_required_stop_distance = stopping_distance(current_speed, FULL_BRAKE_DECEL)

                        set_warning_output(True)
                        stop_drive_motor()
                        schedule_relay_cut(0.075)
                        _pending_latency_t = time.perf_counter()   # τ_trigger for latency
                        set_servo_deg(SERVO_AEB_FULL_DEG)

            elif state == "PARTIAL":
                set_warning_output(True)
                stop_drive_motor()

                brake_level = (SERVO_AEB_PARTIAL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)
                set_servo_deg(SERVO_AEB_PARTIAL_DEG)

                current_speed = wheel_speed_mps  # real encoder, not simulation

                ego.set_speed(current_speed)
                robot_z += current_speed * DT
                virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance else 0.0
                ttc = ttc_from(virtual_distance, current_speed)
                status = "PARTIAL"

                if virtual_distance <= 0.0 and current_speed > STOP_EPS:
                    state = "CRASH"
                    current_speed = 0.0
                    ego.set_speed(0.0)
                    virtual_distance = 0.0
                    brake_level = (SERVO_AEB_FULL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)
                    set_servo_deg(SERVO_AEB_FULL_DEG)
                    stop_drive_motor()
                # encoder reads 0 as soon as motor dies (shaft encoder, not wheel).
                # hold brake until BTN3 reset — do NOT release on encoder zero.

            elif state == "EMERGENCY":
                set_warning_output(True)
                stop_drive_motor()

                brake_level = (SERVO_AEB_FULL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)
                set_servo_deg(SERVO_AEB_FULL_DEG)

                current_speed = wheel_speed_mps  # real encoder, not simulation

                ego.set_speed(current_speed)
                robot_z += current_speed * DT
                virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance else 0.0
                ttc = ttc_from(virtual_distance, current_speed)
                status = "EMERGENCY"

                if virtual_distance <= 0.0 and current_speed > STOP_EPS:
                    state = "CRASH"
                    current_speed = 0.0
                    ego.set_speed(0.0)
                    virtual_distance = 0.0
                    brake_level = (SERVO_AEB_FULL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)
                    set_servo_deg(SERVO_AEB_FULL_DEG)
                    stop_drive_motor()
                # encoder reads 0 as soon as motor dies (shaft encoder, not wheel).
                # hold brake until BTN3 reset — do NOT release on encoder zero.

            elif state == "STOP":
                current_speed = 0.0
                ego.set_speed(0.0)

                warning_on = False
                brake_on = False
                brake_level = 0.0

                set_warning_output(False)
                set_servo_deg(SERVO_AEB_RELEASE_DEG)
                stop_drive_motor()

                ttc = math.inf
                status = "STOP"

            elif state == "CRASH":
                current_speed = 0.0
                ego.set_speed(0.0)

                warning_on = True
                brake_on = True
                brake_level = (SERVO_AEB_FULL_DEG - SERVO_AEB_RELEASE_DEG) / (100.0 - SERVO_AEB_RELEASE_DEG)

                set_warning_output(True)
                set_servo_deg(SERVO_AEB_FULL_DEG)
                stop_drive_motor()

                virtual_distance = 0.0
                ttc = math.inf
                status = "CRASH_RISK"

            timings["logic"] = time.perf_counter() - t0

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------
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

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------
            t0 = time.perf_counter()

            if should_display and cam_frame is not None:
                world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

                cv2.putText(world, f"STATE: {state}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(world, f"Encoder speed: {current_speed:.2f} m/s   RPM: {arduino_rpm:.1f}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                cv2.putText(world, f"Target speed: {manual_speed_mps:.2f} m/s   RPM tgt: {_rpm_target:.0f}  PWM: {_rpm_pwm_actual}",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

                if locked_initial_distance is not None:
                    cv2.putText(world, f"Initial d0: {locked_initial_distance:.2f} m",
                                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                dist_text = f"{virtual_distance:.2f} m" if virtual_distance is not None else "n/a"
                ttc_text = f"{ttc:.2f} s" if math.isfinite(ttc) else "inf"

                cv2.putText(world, f"Virtual dist: {dist_text}   TTC: {ttc_text}",
                            (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                req_text = f"{brake_required_stop_distance:.2f} m" if state in ("PARTIAL", "EMERGENCY") else "n/a"
                cv2.putText(world, f"Stop dist: {req_text}   Brake: {'ON' if brake_on else 'OFF'}",
                            (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                override_color = (0, 0, 255) if manual_override else (255, 255, 0)
                override_txt = (
                    f"PEDAL {pedal_percent:.1f}% ADC={pedal_adc} V={pedal_voltage:.2f} "
                    f"override={int(manual_override)}"
                )
                cv2.putText(world, override_txt,
                            (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.50, override_color, 2)

                relay_txt = "Relay CUT" if manual_override else "Relay normal"
                cv2.putText(world, relay_txt,
                            (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.50, override_color, 2)

                span_m = max(locked_initial_distance if locked_initial_distance else 5.0, 5.0)
                display_span = WORLD_WIDTH - 140
                left_x = 60

                robot_x = int(left_x + min(max((robot_z / span_m) * display_span, 0.0), display_span))
                object_x = (
                    int(left_x + min(max((locked_initial_distance / span_m) * display_span, 0.0), display_span))
                    if locked_initial_distance else WORLD_WIDTH - 80
                )

                cv2.rectangle(
                    world,
                    (robot_x, ROBOT_Y - BOX_H // 2),
                    (robot_x + BOX_W, ROBOT_Y + BOX_H // 2),
                    (0, 255, 0) if state != "CRASH" else (0, 0, 255),
                    -1,
                )
                cv2.rectangle(
                    world,
                    (object_x, OBJECT_Y - BOX_H // 2),
                    (object_x + BOX_W, OBJECT_Y + BOX_H // 2),
                    (0, 0, 255),
                    -1,
                )
                cv2.line(world, (robot_x + BOX_W, ROBOT_Y), (object_x, OBJECT_Y), (255, 255, 255), 2)

                state_labels = {
                    "CRASH": ("COLLISION OCCURRED", (0, 0, 255)),
                    "STOP": ("STOPPED SAFELY", (0, 255, 0)),
                    "PARTIAL": ("PARTIAL BRAKING", (0, 255, 255)),
                    "EMERGENCY": ("EMERGENCY BRAKING", (0, 0, 255)),
                    "FCW": ("FORWARD COLLISION WARNING", (0, 255, 255)),
                }

                if status == "WAIT_DISTANCE":
                    cv2.putText(
                        world,
                        "MOTOR RUNNING - WAITING FOR CAMERA DISTANCE",
                        (520, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                elif state in state_labels:
                    txt, col = state_labels[state]
                    cv2.putText(world, txt, (520, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

                _ds.update({
                    "state": state, "current_speed": current_speed,
                    "virtual_distance": virtual_distance, "ttc": ttc,
                    "brake_on": brake_on, "warning_on": warning_on,
                    "brake_level": brake_level, "manual_override": manual_override,
                    "manual_speed_mps": manual_speed_mps, "pedal_percent": pedal_percent,
                    "time_log": time_log, "speed_log": speed_log,
                    "locked_initial_distance": locked_initial_distance, "robot_z": robot_z,
                })
                _show(cam_frame, world)

            timings["render"] = time.perf_counter() - t0

            # ------------------------------------------------
            # Plot once on STOP / CRASH
            # ------------------------------------------------
            if not plots_shown and state in ("STOP", "CRASH"):
                plots_shown = True
                plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log)

            if loop_counter % PRINT_EVERY_N_LOOPS == 0:
                print_timing(state, timings)

            time.sleep(max(0.0, DT - (time.perf_counter() - loop_t0)))

    except KeyboardInterrupt:
        pass

    finally:
        _motor_ctrl_stop.set()
        _motor_ctrl_thread.join(timeout=0.5)
        _vis.running = False
        stop_drive_motor()
        cv2.destroyAllWindows()
        cleanup_gpio()

        if time_log and not plots_shown:
            plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log)


if __name__ == "__main__":
    main()
