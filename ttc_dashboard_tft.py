import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

import board
import digitalio
import adafruit_rgb_display.ili9341 as ili9341

from ego_sim import EgoVehicle
from webcam_distance_test import main

# ============================================================
# TFT CONFIG
# ============================================================
# Wiring assumption:
#   TFT SCK  -> GPIO11 / SCLK
#   TFT MOSI -> GPIO10 / MOSI
#   TFT MISO -> GPIO9  / MISO (usually unused by the display)
#   TFT CS   -> GPIO8  / CE0
#   TFT DC   -> GPIO18
#   TFT RST  -> GPIO23
#
# If your board wiring is different, change these pins accordingly.
DC_PIN = board.D18
RST_PIN = board.D23
CS_PIN = board.CE0

TFT_ROTATION = 0   # try 90 or 270 if the screen is sideways
SPI_BAUDRATE = 24000000

spi = board.SPI()
while not spi.try_lock():
    pass
spi.configure(baudrate=SPI_BAUDRATE)
spi.unlock()

cs = digitalio.DigitalInOut(CS_PIN)
dc = digitalio.DigitalInOut(DC_PIN)
rst = digitalio.DigitalInOut(RST_PIN)

disp = ili9341.ILI9341(
    spi,
    cs=cs,
    dc=dc,
    rst=rst,
    rotation=TFT_ROTATION,
    baudrate=SPI_BAUDRATE,
)

TFT_WIDTH = disp.width
TFT_HEIGHT = disp.height

# ============================================================
# DISPLAY OPTIONS
# ============================================================
UPDATE_TFT_EVERY_N_FRAMES = 5
SHOW_DESKTOP_CAMERA = True
SHOW_DESKTOP_PLOTS = True

# ============================================================
# Timing / debug
# ============================================================
DEBUG_TIMING = True
PRINT_EVERY_N_FRAMES = 1

frame_counter = 0

def _ms(seconds: float) -> float:
    return seconds * 1000.0

def print_timing(label: str, timings: dict):
    if not DEBUG_TIMING:
        return
    parts = " | ".join(f"{k}={_ms(v):.2f}ms" for k, v in timings.items())
    total = sum(timings.values())
    print(f"[{label}] {parts} | total={_ms(total):.2f}ms", flush=True)

# ============================================================
# Config
# ============================================================
SCALE = 10
WORLD_WIDTH = 1000
WORLD_HEIGHT = 260

MIN_VALID_DISTANCE = 0.10

# Init behavior
INIT_REQUIRED_SAMPLES = 8
INIT_MAX_WAIT_SEC = 2.0

# ============================================================
# TTC / braking settings
# ============================================================
DT = 0.02

SAFE_TTC = 2.6
FCW_TTC = 1.6
PARTIAL_TTC = 0.6

FULL_BRAKE_DECEL = 6.43
PARTIAL_BRAKE_DECEL = FULL_BRAKE_DECEL * 0.4

STOP_EPS = 0.01

# Demo speed controls if encoder is not connected yet
USE_MANUAL_SPEED = True
MANUAL_SPEED_STEP = 0.10
MAX_DEMO_SPEED = 5.0

# ============================================================
# Fonts
# ============================================================
def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass

    return ImageFont.load_default()

FONT_STATE = load_font(26, bold=True)
FONT_LABEL = load_font(14, bold=True)
FONT_VALUE = load_font(28, bold=True)
FONT_SMALL = load_font(12, bold=False)
FONT_MED = load_font(16, bold=True)

# ============================================================
# Helpers
# ============================================================
def depth_to_meters(depth_value: float) -> float:
    return max(float(depth_value) * SCALE, 0.0)

def ttc_from(distance_m: float, speed_mps: float) -> float:
    if speed_mps <= 1e-6:
        return math.inf
    return distance_m / speed_mps

def stopping_distance(speed_mps: float, decel_mps2: float) -> float:
    if speed_mps <= 1e-6 or decel_mps2 <= 1e-6:
        return 0.0
    return (speed_mps * speed_mps) / (2.0 * decel_mps2)

def ttc_status(ttc: float) -> str:
    if not math.isfinite(ttc):
        return "SAFE"
    if ttc > SAFE_TTC:
        return "SAFE"
    if ttc > FCW_TTC:
        return "FCW"
    if ttc > PARTIAL_TTC:
        return "PARTIAL"
    return "EMERGENCY"

def state_color(state: str):
    if state in ("SAFE", "RUN"):
        return (20, 130, 40)
    if state == "FCW":
        return (190, 160, 0)
    if state == "PARTIAL":
        return (190, 110, 0)
    if state == "EMERGENCY":
        return (180, 0, 0)
    if state == "STOP":
        return (0, 140, 70)
    if state == "CRASH":
        return (180, 0, 0)
    return (60, 60, 60)

def draw_centered_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) / 2
    y = y0 + (y1 - y0 - th) / 2 - 1
    draw.text((x, y), text, font=font, fill=fill)

def draw_metric_card(draw, box, title, value, unit, accent, value_font=FONT_VALUE):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, outline=accent, width=2, fill=(18, 18, 18))
    draw.text((x0 + 10, y0 + 8), title, font=FONT_LABEL, fill=accent)

    value_text = f"{value}"
    draw.text((x0 + 10, y0 + 24), value_text, font=value_font, fill=(255, 255, 255))

    if unit:
        bbox = draw.textbbox((0, 0), value_text, font=value_font)
        vw = bbox[2] - bbox[0]
        draw.text((x0 + 10 + vw + 6, y0 + 42), unit, font=FONT_SMALL, fill=(200, 200, 200))

def draw_bar(draw, box, title, level, on_text, accent, bg_fill=(40, 40, 40)):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, outline=accent, width=2, fill=(18, 18, 18))
    draw.text((x0 + 10, y0 + 6), title, font=FONT_LABEL, fill=accent)

    bar_x0 = x0 + 10
    bar_y0 = y0 + 32
    bar_x1 = x1 - 10
    bar_y1 = y0 + 56

    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=8, fill=bg_fill)

    level = float(np.clip(level, 0.0, 1.0))
    fill_x1 = bar_x0 + int((bar_x1 - bar_x0) * level)
    if fill_x1 > bar_x0:
        draw.rounded_rectangle((bar_x0, bar_y0, fill_x1, bar_y1), radius=8, fill=accent)

    draw.text((x0 + 10, y0 + 60), on_text, font=FONT_SMALL, fill=(230, 230, 230))

def render_dashboard(
    state: str,
    speed_mps: float,
    virtual_distance_m: float | None,
    ttc_value: float,
    brake_level: float,
    brake_on: bool,
    warning_on: bool,
    locked_initial_distance: float | None,
    live_distance: float | None,
    manual_speed_mps: float,
):
    img = Image.new("RGB", (TFT_WIDTH, TFT_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    banner_color = state_color(state)
    draw.rounded_rectangle((8, 8, TFT_WIDTH - 8, 58), radius=14, fill=banner_color)

    draw_centered_text(draw, (8, 8, TFT_WIDTH - 8, 38), state, FONT_STATE, (255, 255, 255))
    mode_text = f"Brake {'ON' if brake_on else 'OFF'}  |  Warning {'ON' if warning_on else 'OFF'}"
    draw_centered_text(draw, (8, 34, TFT_WIDTH - 8, 58), mode_text, FONT_SMALL, (245, 245, 245))

    card_w = TFT_WIDTH - 16
    card_x = 8

    speed_box = (card_x, 68, card_x + card_w, 125)
    ttc_box = (card_x, 132, card_x + card_w, 189)
    dist_box = (card_x, 196, card_x + card_w, 253)

    speed_val = f"{speed_mps:0.2f}"
    ttc_val = "inf" if not math.isfinite(ttc_value) else f"{ttc_value:0.2f}"
    dist_val = "n/a" if virtual_distance_m is None else f"{virtual_distance_m:0.2f}"

    draw_metric_card(draw, speed_box, "SPEED", speed_val, "m/s", (80, 170, 255))
    draw_metric_card(draw, ttc_box, "TTC", ttc_val, "s", state_color(ttc_status(ttc_value)))
    draw_metric_card(draw, dist_box, "DISTANCE", dist_val, "m", (180, 180, 180))

    bar_box = (8, 260, TFT_WIDTH - 8, 300)
    brake_text = f"{brake_level * 100:0.0f}%"
    draw_bar(draw, bar_box, "BRAKE", brake_level, f"LEVEL {brake_text}", (220, 60, 60))

    footer_y = 303
    footer_left = f"d0 {locked_initial_distance:0.2f}m" if locked_initial_distance is not None else "d0 n/a"
    footer_mid = f"live {live_distance:0.2f}m" if live_distance is not None else "live n/a"
    footer_right = f"man {manual_speed_mps:0.2f}"

    draw.text((8, footer_y), footer_left, font=FONT_SMALL, fill=(200, 200, 200))
    draw.text((84, footer_y), footer_mid, font=FONT_SMALL, fill=(200, 200, 200))
    draw.text((174, footer_y), footer_right, font=FONT_SMALL, fill=(200, 200, 200))

    return img

def display_on_tft(img: Image.Image):
    disp.image(img)

def set_warning_output(enabled: bool):
    pass

def set_brake_output(level: float):
    level = float(np.clip(level, 0.0, 1.0))
    pass

def reset_ego(ego: EgoVehicle):
    if hasattr(ego, "reset") and callable(getattr(ego, "reset")):
        ego.reset()
    else:
        if hasattr(ego, "position"):
            ego.position = 0.0
        if hasattr(ego, "z"):
            ego.z = 0.0
        if hasattr(ego, "velocity"):
            ego.velocity = 0.0
    ego.set_speed(0.0)

def get_closest_live_distance(tracks) -> float | None:
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

def first_state_idx(state_log, target):
    for i, s in enumerate(state_log):
        if s == target:
            return i
    return None

def plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log):
    if not time_log or not SHOW_DESKTOP_PLOTS:
        return

    fcw_idx = first_state_idx(state_log, "FCW")
    partial_idx = first_state_idx(state_log, "PARTIAL")
    emergency_idx = first_state_idx(state_log, "EMERGENCY")
    stop_idx = first_state_idx(state_log, "STOP")
    crash_idx = first_state_idx(state_log, "CRASH")

    plt.figure(figsize=(9, 4))
    plt.plot(time_log, distance_log, label="Remaining distance")
    if fcw_idx is not None:
        plt.axvline(time_log[fcw_idx], linestyle="--", label="FCW trigger")
    if partial_idx is not None:
        plt.axvline(time_log[partial_idx], linestyle="--", label="Partial brake trigger")
    if emergency_idx is not None:
        plt.axvline(time_log[emergency_idx], linestyle="--", label="Emergency brake trigger")
    if stop_idx is not None:
        plt.axvline(time_log[stop_idx], linestyle=":", label="Stop")
    if crash_idx is not None:
        plt.axvline(time_log[crash_idx], linestyle=":", label="Crash")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distance vs Time")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(9, 4))
    plt.plot(time_log, speed_log, label="Speed")
    if fcw_idx is not None:
        plt.axvline(time_log[fcw_idx], linestyle="--", label="FCW trigger")
    if partial_idx is not None:
        plt.axvline(time_log[partial_idx], linestyle="--", label="Partial brake trigger")
    if emergency_idx is not None:
        plt.axvline(time_log[emergency_idx], linestyle="--", label="Emergency brake trigger")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed vs Time")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(9, 4))
    ttc_arr = np.array(ttc_log, dtype=float)
    finite_ttc = ttc_arr[np.isfinite(ttc_arr)]
    if finite_ttc.size > 0:
        y_top = max(float(np.max(finite_ttc)), SAFE_TTC + 0.5)
    else:
        y_top = SAFE_TTC + 1.0

    plt.axhspan(0, PARTIAL_TTC, color="red", alpha=0.20, label="Emergency")
    plt.axhspan(PARTIAL_TTC, FCW_TTC, color="orange", alpha=0.20, label="Partial")
    plt.axhspan(FCW_TTC, SAFE_TTC, color="yellow", alpha=0.20, label="FCW")
    plt.axhspan(SAFE_TTC, y_top, color="green", alpha=0.12, label="Safe")

    plt.axhline(SAFE_TTC, linestyle="--", color="blue")
    plt.axhline(FCW_TTC, linestyle="--", color="blue")
    plt.axhline(PARTIAL_TTC, linestyle="--", color="blue")

    plt.plot(time_log, ttc_log, color="black", label="TTC")

    if fcw_idx is not None:
        plt.axvline(time_log[fcw_idx], linestyle="--", label="FCW trigger")
    if partial_idx is not None:
        plt.axvline(time_log[partial_idx], linestyle="--", label="Partial brake trigger")
    if emergency_idx is not None:
        plt.axvline(time_log[emergency_idx], linestyle="--", label="Emergency brake trigger")

    plt.xlabel("Time (s)")
    plt.ylabel("TTC (s)")
    plt.title("TTC vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Demo speed fallback
# ============================================================
manual_speed_mps = 0.0

def read_wheel_speed_mps() -> float:
    return max(manual_speed_mps, 0.0)

# ============================================================
# Init
# ============================================================
ego = EgoVehicle()
ego.set_speed(0.0)

state = "IDLE"

locked_initial_distance = None
virtual_distance = None
robot_z = 0.0
last_frame_time = None

init_samples = []
init_start_time = None

brake_on = False
warning_on = False
last_live_distance = None
plots_shown = False

brake_trigger_speed = 0.0
brake_required_stop_distance = 0.0
brake_trigger_mode = None

time_log = []
distance_log = []
speed_log = []
ttc_log = []
travel_log = []
stop_req_log = []
state_log = []
sim_time = 0.0

current_speed = 0.0
brake_level = 0.0

ttc = math.inf
status = "SAFE"

# ============================================================
# Main loop
# ============================================================
for frame, tracks in main(yield_every_frame=True):
    frame_counter += 1
    loop_t0 = time.perf_counter()
    timings = {}

    # dt
    t0 = time.perf_counter()
    now = time.perf_counter()
    dt = DT
    last_frame_time = now
    timings["dt"] = time.perf_counter() - t0

    # Keyboard controls
    t0 = time.perf_counter()
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        timings["key"] = time.perf_counter() - t0
        print_timing("EXIT", timings)
        break

    elif key == ord("i"):
        if state in ("IDLE", "STOP", "CRASH"):
            state = "INIT"
            init_samples = []
            init_start_time = now
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
            set_warning_output(False)
            set_brake_output(0.0)
            reset_ego(ego)

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
        set_warning_output(False)
        set_brake_output(0.0)
        reset_ego(ego)

    elif USE_MANUAL_SPEED:
        if key == ord("w"):
            manual_speed_mps = min(manual_speed_mps + MANUAL_SPEED_STEP, MAX_DEMO_SPEED)
        elif key == ord("s"):
            manual_speed_mps = max(manual_speed_mps - MANUAL_SPEED_STEP, 0.0)
        elif key == ord(" "):
            manual_speed_mps = 0.0

    timings["key"] = time.perf_counter() - t0

    # Optional camera preview
    t0 = time.perf_counter()
    if SHOW_DESKTOP_CAMERA:
        cv2.imshow("CV + Tracking", frame)
    timings["camera_imshow"] = time.perf_counter() - t0

    # Get live distance
    t0 = time.perf_counter()
    live_distance = get_closest_live_distance(tracks)
    last_live_distance = live_distance if live_distance is not None else last_live_distance
    timings["live_distance"] = time.perf_counter() - t0

    # ========================================================
    # IDLE
    # ========================================================
    if state == "IDLE":
        t0 = time.perf_counter()

        dashboard = render_dashboard(
            state=state,
            speed_mps=current_speed,
            virtual_distance_m=virtual_distance,
            ttc_value=ttc,
            brake_level=brake_level,
            brake_on=brake_on,
            warning_on=warning_on,
            locked_initial_distance=locked_initial_distance,
            live_distance=last_live_distance,
            manual_speed_mps=manual_speed_mps,
        )

        if frame_counter % UPDATE_TFT_EVERY_N_FRAMES == 0:
            display_on_tft(dashboard)

        timings["idle_render"] = time.perf_counter() - t0
        timings["loop_total"] = time.perf_counter() - loop_t0
        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_timing("IDLE", timings)
        continue

    # ========================================================
    # INIT
    # ========================================================
    if state == "INIT":
        t0 = time.perf_counter()

        if live_distance is not None:
            init_samples.append(live_distance)

        if (len(init_samples) >= INIT_REQUIRED_SAMPLES) or (
            init_start_time is not None and (now - init_start_time) >= INIT_MAX_WAIT_SEC and len(init_samples) > 0
        ):
            locked_initial_distance = float(np.mean(init_samples))
            virtual_distance = locked_initial_distance
            robot_z = 0.0
            current_speed = 0.0
            brake_on = False
            warning_on = False
            brake_level = 0.0
            set_warning_output(False)
            set_brake_output(0.0)
            reset_ego(ego)
            ego.set_speed(0.0)
            state = "RUN"

        dashboard = render_dashboard(
            state="INIT",
            speed_mps=current_speed,
            virtual_distance_m=virtual_distance,
            ttc_value=ttc,
            brake_level=brake_level,
            brake_on=brake_on,
            warning_on=warning_on,
            locked_initial_distance=locked_initial_distance,
            live_distance=live_distance,
            manual_speed_mps=manual_speed_mps,
        )

        if frame_counter % UPDATE_TFT_EVERY_N_FRAMES == 0:
            display_on_tft(dashboard)

        timings["init_render"] = time.perf_counter() - t0
        timings["loop_total"] = time.perf_counter() - loop_t0
        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_timing("INIT", timings)
        continue

    # ========================================================
    # RUN / FCW / PARTIAL / EMERGENCY / STOP / CRASH
    # ========================================================
    t0 = time.perf_counter()
    wheel_speed_mps = read_wheel_speed_mps()
    timings["read_speed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if locked_initial_distance is not None:
        virtual_distance = max(locked_initial_distance - robot_z, 0.0)
    else:
        virtual_distance = 0.0
    timings["distance_calc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if state in ("RUN", "FCW"):
        current_speed = wheel_speed_mps
        ego.set_speed(current_speed)

        ttc = ttc_from(virtual_distance, current_speed)
        status = ttc_status(ttc)

        if status == "SAFE":
            state = "RUN"
            warning_on = False
            brake_on = False
            brake_level = 0.0
            set_warning_output(False)
            set_brake_output(0.0)
            robot_z += current_speed * dt

        elif status == "FCW":
            state = "FCW"
            warning_on = True
            brake_on = False
            brake_level = 0.0
            set_warning_output(True)
            set_brake_output(0.0)
            robot_z += current_speed * dt

        elif status == "PARTIAL":
            state = "PARTIAL"
            warning_on = True
            brake_on = True
            brake_level = PARTIAL_BRAKE_DECEL / FULL_BRAKE_DECEL
            brake_trigger_speed = current_speed
            brake_trigger_mode = "PARTIAL"
            brake_required_stop_distance = stopping_distance(brake_trigger_speed, PARTIAL_BRAKE_DECEL)
            set_warning_output(True)
            set_brake_output(brake_level)

        else:
            state = "EMERGENCY"
            warning_on = True
            brake_on = True
            brake_level = 1.0
            brake_trigger_speed = current_speed
            brake_trigger_mode = "EMERGENCY"
            brake_required_stop_distance = stopping_distance(brake_trigger_speed, FULL_BRAKE_DECEL)
            set_warning_output(True)
            set_brake_output(brake_level)

    elif state == "PARTIAL":
        warning_on = True
        brake_on = True
        brake_level = PARTIAL_BRAKE_DECEL / FULL_BRAKE_DECEL
        set_warning_output(True)
        set_brake_output(brake_level)

        current_speed = max(current_speed - PARTIAL_BRAKE_DECEL * dt, 0.0)
        ego.set_speed(current_speed)
        robot_z += current_speed * dt

        virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance is not None else 0.0
        ttc = ttc_from(virtual_distance, current_speed)
        status = "PARTIAL"

        if virtual_distance <= 0.0 and current_speed > STOP_EPS:
            state = "CRASH"
            current_speed = 0.0
            ego.set_speed(0.0)
            virtual_distance = 0.0
            brake_level = 1.0
            set_brake_output(brake_level)

        elif current_speed <= STOP_EPS:
            state = "STOP"
            current_speed = 0.0
            ego.set_speed(0.0)
            brake_on = True
            brake_level = PARTIAL_BRAKE_DECEL / FULL_BRAKE_DECEL
            set_brake_output(brake_level)

    elif state == "EMERGENCY":
        warning_on = True
        brake_on = True
        brake_level = 1.0
        set_warning_output(True)
        set_brake_output(brake_level)

        current_speed = max(current_speed - FULL_BRAKE_DECEL * dt, 0.0)
        ego.set_speed(current_speed)
        robot_z += current_speed * dt

        virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance is not None else 0.0
        ttc = ttc_from(virtual_distance, current_speed)
        status = "EMERGENCY"

        if virtual_distance <= 0.0 and current_speed > STOP_EPS:
            state = "CRASH"
            current_speed = 0.0
            ego.set_speed(0.0)
            virtual_distance = 0.0
            brake_level = 1.0
            set_brake_output(brake_level)

        elif current_speed <= STOP_EPS:
            state = "STOP"
            current_speed = 0.0
            ego.set_speed(0.0)
            brake_on = True
            brake_level = 1.0
            set_brake_output(0.0)

    elif state == "STOP":
        current_speed = 0.0
        ego.set_speed(0.0)
        warning_on = False
        brake_on = True
        brake_level = 0.0
        set_warning_output(False)
        set_brake_output(0.0)
        ttc = math.inf
        status = "STOP"

    elif state == "CRASH":
        current_speed = 0.0
        ego.set_speed(0.0)
        warning_on = True
        brake_on = True
        brake_level = 1.0
        set_warning_output(True)
        set_brake_output(brake_level)
        virtual_distance = 0.0
        ttc = math.inf
        status = "CRASH_RISK"

    timings["logic"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if state not in ("RUN", "FCW", "PARTIAL", "EMERGENCY"):
        if not math.isfinite(ttc):
            ttc = math.inf
        if virtual_distance is None:
            virtual_distance = 0.0
    timings["sanity"] = time.perf_counter() - t0

    # Logs
    t0 = time.perf_counter()
    sim_time += DT
    time_log.append(sim_time)
    distance_log.append(virtual_distance if virtual_distance is not None else 0.0)
    speed_log.append(current_speed)
    ttc_log.append(ttc if math.isfinite(ttc) else np.nan)
    travel_log.append(robot_z)

    if state in ("PARTIAL", "EMERGENCY"):
        stop_req_log.append(brake_required_stop_distance)
    else:
        stop_req_log.append(0.0)

    state_log.append(state)
    timings["logging"] = time.perf_counter() - t0

    # TFT dashboard render
    t0 = time.perf_counter()
    dashboard = render_dashboard(
        state=state,
        speed_mps=current_speed,
        virtual_distance_m=virtual_distance,
        ttc_value=ttc,
        brake_level=brake_level,
        brake_on=brake_on,
        warning_on=warning_on,
        locked_initial_distance=locked_initial_distance,
        live_distance=live_distance,
        manual_speed_mps=manual_speed_mps,
    )

    if frame_counter % UPDATE_TFT_EVERY_N_FRAMES == 0:
        display_on_tft(dashboard)

    timings["render"] = time.perf_counter() - t0

    # Plot once when run ends
    t0 = time.perf_counter()
    if not plots_shown and state in ("STOP", "CRASH"):
        plots_shown = True
        plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log)
    timings["plot"] = time.perf_counter() - t0

    timings["loop_total"] = time.perf_counter() - loop_t0
    if frame_counter % PRINT_EVERY_N_FRAMES == 0:
        print_timing(state, timings)

cv2.destroyAllWindows()

if time_log and not plots_shown:
    plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log)