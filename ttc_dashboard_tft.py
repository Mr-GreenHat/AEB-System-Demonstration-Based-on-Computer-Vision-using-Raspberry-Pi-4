import cv2
import numpy as np
import time
import math
import os

from PIL import Image, ImageDraw, ImageFont

import board
import digitalio
import adafruit_rgb_display.ili9341 as ili9341

from ego_sim import EgoVehicle
from webcam_distance_test import main

# ============================================================
# TFT CONFIG
# ============================================================
DC_PIN = board.D18
RST_PIN = board.D23
CS_PIN = board.CE0

TFT_ROTATION = 0         # try 90 or 270 if sideways
SPI_BAUDRATE = 24000000

spi = board.SPI()
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

GRAPH_DELAY_SEC = 2.0
GRAPH_ANIM_SEC = 2.0

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
# CONFIG
# ============================================================
SCALE = 10
WORLD_WIDTH = 1000
WORLD_HEIGHT = 260

MIN_VALID_DISTANCE = 0.10

INIT_REQUIRED_SAMPLES = 8
INIT_MAX_WAIT_SEC = 2.0

DT = 0.02

SAFE_TTC = 2.6
FCW_TTC = 1.6
PARTIAL_TTC = 0.6

FULL_BRAKE_DECEL = 6.43
PARTIAL_BRAKE_DECEL = FULL_BRAKE_DECEL * 0.4

STOP_EPS = 0.01

USE_MANUAL_SPEED = True
MANUAL_SPEED_STEP = 0.10
MAX_DEMO_SPEED = 5.0

# ============================================================
# FONTS
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
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass

    return ImageFont.load_default()

FONT_TITLE = load_font(22, bold=True)
FONT_STATE = load_font(26, bold=True)
FONT_LABEL = load_font(14, bold=True)
FONT_VALUE = load_font(28, bold=True)
FONT_SMALL = load_font(12, bold=False)
FONT_MED = load_font(16, bold=True)
FONT_LARGE = load_font(24, bold=True)

# ============================================================
# HELPERS
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

def zone_color_from_ttc(ttc: float):
    if not math.isfinite(ttc):
        return (20, 130, 40)
    if ttc > SAFE_TTC:
        return (20, 130, 40)
    if ttc > FCW_TTC:
        return (190, 160, 0)
    if ttc > PARTIAL_TTC:
        return (190, 110, 0)
    return (180, 0, 0)

def zone_label_from_ttc(ttc: float):
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
    if state in ("EMERGENCY", "CRASH"):
        return (180, 0, 0)
    if state == "STOP":
        return (0, 140, 70)
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

def display_image(img: Image.Image):
    img = img.convert("RGB")
    disp.image(img)

# ============================================================
# TFT RENDERERS
# ============================================================
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
    draw_metric_card(draw, ttc_box, "TTC", ttc_val, "s", zone_color_from_ttc(ttc_value))
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

def render_stop_hold_screen(result_label: str, distance_left: float, countdown: float):
    img = Image.new("RGB", (TFT_WIDTH, TFT_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((8, 8, TFT_WIDTH - 8, 60), radius=14, fill=state_color(result_label))
    draw_centered_text(draw, (8, 8, TFT_WIDTH - 8, 60), result_label, FONT_STATE, (255, 255, 255))

    draw.rounded_rectangle((8, 78, TFT_WIDTH - 8, 176), radius=14, outline=(90, 90, 90), width=2, fill=(18, 18, 18))
    draw.text((18, 90), "Graph starts in", font=FONT_MED, fill=(220, 220, 220))
    draw.text((18, 122), f"{countdown:0.1f} s", font=FONT_LARGE, fill=(255, 255, 255))
    draw.text((18, 152), "Hold still", font=FONT_SMALL, fill=(180, 180, 180))

    draw.rounded_rectangle((8, 194, TFT_WIDTH - 8, 258), radius=14, outline=(120, 120, 120), width=2, fill=(15, 15, 15))
    draw.text((18, 208), f"Distance left: {distance_left:0.2f} m", font=FONT_MED, fill=(255, 255, 255))

    draw.rounded_rectangle((8, 270, TFT_WIDTH - 8, 300), radius=12, outline=(80, 80, 80), width=1, fill=(18, 18, 18))
    draw.text((18, 278), "Preparing animated speed graph", font=FONT_SMALL, fill=(210, 210, 210))

    return img

def render_animated_speed_graph(
    time_log,
    speed_log,
    ttc_log,
    distance_left: float,
    progress: float,
    final_label: str,
):
    img = Image.new("RGB", (TFT_WIDTH, TFT_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin_x = 10

    # Title
    draw.rounded_rectangle((8, 6, TFT_WIDTH - 8, 36), radius=10, fill=state_color(final_label))
    draw_centered_text(draw, (8, 6, TFT_WIDTH - 8, 36), f"{final_label}  |  SPEED vs TIME", FONT_SMALL, (255, 255, 255))

    if not time_log or len(time_log) < 2:
        draw.text((18, 60), "No graph data yet", font=FONT_MED, fill=(255, 255, 255))
        draw.text((18, 90), f"Distance left: {distance_left:0.2f} m", font=FONT_MED, fill=(255, 255, 255))
        return img

    max_time = max(float(time_log[-1]), 1e-6)
    max_speed = max(1.0, float(max(speed_log)) * 1.15)

    progress = float(np.clip(progress, 0.0, 1.0))
    visible_count = max(2, min(len(time_log), int(math.ceil(len(time_log) * progress))))
    visible_times = time_log[:visible_count]
    visible_speeds = speed_log[:visible_count]
    visible_ttcs = ttc_log[:visible_count]
    visible_time_end = visible_times[-1]

    # TTC zone strip
    strip_x0 = margin_x
    strip_y0 = 42
    strip_x1 = TFT_WIDTH - margin_x
    strip_y1 = 58
    draw.rounded_rectangle((strip_x0, strip_y0, strip_x1, strip_y1), radius=6, outline=(80, 80, 80), width=1, fill=(25, 25, 25))

    for i in range(1, len(visible_times)):
        t0 = visible_times[i - 1]
        t1 = visible_times[i]
        x0 = strip_x0 + int((t0 / max_time) * (strip_x1 - strip_x0))
        x1 = strip_x0 + int((t1 / max_time) * (strip_x1 - strip_x0))
        color = zone_color_from_ttc(visible_ttcs[i - 1])
        if x1 <= x0:
            x1 = x0 + 1
        draw.rectangle((x0, strip_y0 + 1, x1, strip_y1 - 1), fill=color)

    if visible_time_end < max_time:
        x_future = strip_x0 + int((visible_time_end / max_time) * (strip_x1 - strip_x0))
        draw.rectangle((x_future, strip_y0 + 1, strip_x1, strip_y1 - 1), fill=(40, 40, 40))

    # Graph area
    gx0 = 18
    gy0 = 72
    gx1 = TFT_WIDTH - 12
    gy1 = 236

    draw.rounded_rectangle((gx0 - 4, gy0 - 4, gx1 + 4, gy1 + 4), radius=10, outline=(80, 80, 80), width=2, fill=(10, 10, 10))

    # Grid
    for i in range(1, 5):
        y = gy0 + int((gy1 - gy0) * i / 5)
        draw.line((gx0, y, gx1, y), fill=(35, 35, 35), width=1)
    for i in range(1, 5):
        x = gx0 + int((gx1 - gx0) * i / 5)
        draw.line((x, gy0, x, gy1), fill=(35, 35, 35), width=1)

    # Axis labels
    draw.text((gx0, gy1 + 4), "0s", font=FONT_SMALL, fill=(190, 190, 190))
    draw.text((gx1 - 48, gy1 + 4), f"{visible_time_end:0.1f}s/{max_time:0.1f}s", font=FONT_SMALL, fill=(190, 190, 190))
    draw.text((2, gy0 - 2), f"{max_speed:0.1f}", font=FONT_SMALL, fill=(190, 190, 190))
    draw.text((8, gy1 - 10), "0", font=FONT_SMALL, fill=(190, 190, 190))

    # Speed curve with TTC coloring
    points = []
    for t, s in zip(visible_times, visible_speeds):
        x = gx0 + int((t / max_time) * (gx1 - gx0))
        y = gy1 - int((s / max_speed) * (gy1 - gy0))
        points.append((x, y))

    for i in range(1, len(points)):
        color = zone_color_from_ttc(visible_ttcs[i - 1])
        draw.line((points[i - 1], points[i]), fill=color, width=3)

    if points:
        x, y = points[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 255))

    # Cursor
    cursor_x = gx0 + int((visible_time_end / max_time) * (gx1 - gx0))
    draw.line((cursor_x, gy0, cursor_x, gy1), fill=(255, 255, 255), width=1)

    # Legend
    legend_y = 246
    draw.rounded_rectangle((8, 244, TFT_WIDTH - 8, 300), radius=12, outline=(90, 90, 90), width=1, fill=(15, 15, 15))
    draw.text((12, 250), f"Distance left: {distance_left:0.2f} m", font=FONT_MED, fill=(255, 255, 255))

    last_speed = visible_speeds[-1] if visible_speeds else 0.0
    last_ttc = visible_ttcs[-1] if visible_ttcs else math.inf
    last_zone = zone_label_from_ttc(last_ttc)

    draw.text((12, 275), f"Speed: {last_speed:0.2f} m/s", font=FONT_SMALL, fill=(210, 210, 210))
    draw.text((120, 275), f"TTC: {last_zone}", font=FONT_SMALL, fill=zone_color_from_ttc(last_ttc))
    draw.text((180, 275), f"{progress * 100:0.0f}%", font=FONT_SMALL, fill=(210, 210, 210))

    # Footer message
    draw.text((12, 292), "Green=SAFE  Yellow=FCW  Orange=PARTIAL  Red=EMERGENCY", font=FONT_SMALL, fill=(180, 180, 180))

    return img

# ============================================================
# CAMERA / TRACKING HELPERS
# ============================================================
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

# ============================================================
# DEMO SPEED FALLBACK
# ============================================================
manual_speed_mps = 0.0

def read_wheel_speed_mps() -> float:
    return max(manual_speed_mps, 0.0)

# ============================================================
# INIT
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

# Terminal / graph state
terminal_hold_start = None
graph_start_time = None
graph_distance_snapshot = 0.0
graph_result_label = ""

# ============================================================
# MAIN LOOP
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
        if state in ("IDLE", "STOP", "CRASH", "GRAPH"):
            state = "INIT"
            init_samples = []
            init_start_time = now
            locked_initial_distance = None
            virtual_distance = None
            robot_z = 0.0
            brake_on = False
            warning_on = False
            brake_level = 0.0
            terminal_hold_start = None
            graph_start_time = None
            graph_distance_snapshot = 0.0
            graph_result_label = ""
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
        terminal_hold_start = None
        graph_start_time = None
        graph_distance_snapshot = 0.0
        graph_result_label = ""
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

    # Desktop camera preview
    t0 = time.perf_counter()
    if SHOW_DESKTOP_CAMERA:
        cv2.imshow("CV + Tracking", frame)
    timings["camera_imshow"] = time.perf_counter() - t0

    # Live distance
    t0 = time.perf_counter()
    live_distance = get_closest_live_distance(tracks)
    last_live_distance = live_distance if live_distance is not None else last_live_distance
    timings["live_distance"] = time.perf_counter() - t0

    # ========================================================
    # IDLE
    # ========================================================
    if state == "IDLE":
        t0 = time.perf_counter()

        dashboard_speed = manual_speed_mps if USE_MANUAL_SPEED else current_speed
        dashboard = render_dashboard(
            state=state,
            speed_mps=dashboard_speed,
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
            display_image(dashboard)

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

        elapsed = now - init_start_time if init_start_time is not None else 0.0

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
            speed_mps=0.0,
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
            display_image(dashboard)

        timings["init_render"] = time.perf_counter() - t0
        timings["loop_total"] = time.perf_counter() - loop_t0
        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_timing("INIT", timings)
        continue

    # ========================================================
    # RUN / FCW / PARTIAL / EMERGENCY / STOP / CRASH / GRAPH
    # ========================================================
    if state == "GRAPH":
        t0 = time.perf_counter()

        if graph_start_time is None:
            graph_start_time = now

        progress = (now - graph_start_time) / GRAPH_ANIM_SEC
        img = render_animated_speed_graph(
            time_log=time_log,
            speed_log=speed_log,
            ttc_log=ttc_log,
            distance_left=graph_distance_snapshot,
            progress=progress,
            final_label=graph_result_label or "ENDED",
        )
        display_image(img)

        timings["graph_render"] = time.perf_counter() - t0
        timings["loop_total"] = time.perf_counter() - loop_t0
        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_timing("GRAPH", timings)
        continue

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
            if terminal_hold_start is None:
                terminal_hold_start = now
                graph_distance_snapshot = 0.0
                graph_result_label = "COLLISION"

        elif current_speed <= STOP_EPS:
            state = "STOP"
            current_speed = 0.0
            ego.set_speed(0.0)
            brake_on = True
            brake_level = 0.0
            set_brake_output(0.0)
            if terminal_hold_start is None:
                terminal_hold_start = now
                graph_distance_snapshot = virtual_distance
                graph_result_label = "STOPPED SAFELY"

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
            if terminal_hold_start is None:
                terminal_hold_start = now
                graph_distance_snapshot = 0.0
                graph_result_label = "COLLISION"

        elif current_speed <= STOP_EPS:
            state = "STOP"
            current_speed = 0.0
            ego.set_speed(0.0)
            brake_on = True
            brake_level = 0.0
            set_brake_output(0.0)
            if terminal_hold_start is None:
                terminal_hold_start = now
                graph_distance_snapshot = virtual_distance
                graph_result_label = "STOPPED SAFELY"

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

        if terminal_hold_start is None:
            terminal_hold_start = now
            graph_distance_snapshot = virtual_distance if virtual_distance is not None else 0.0
            graph_result_label = "STOPPED SAFELY"

        if (now - terminal_hold_start) >= GRAPH_DELAY_SEC:
            graph_start_time = now
            state = "GRAPH"

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
        status = "CRASH"

        if terminal_hold_start is None:
            terminal_hold_start = now
            graph_distance_snapshot = 0.0
            graph_result_label = "COLLISION"

        if (now - terminal_hold_start) >= GRAPH_DELAY_SEC:
            graph_start_time = now
            state = "GRAPH"

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

    # TFT rendering
    t0 = time.perf_counter()

    if state in ("STOP", "CRASH"):
        remaining = max(GRAPH_DELAY_SEC - (now - terminal_hold_start), 0.0) if terminal_hold_start is not None else GRAPH_DELAY_SEC
        img = render_stop_hold_screen(
            result_label=graph_result_label or state,
            distance_left=graph_distance_snapshot,
            countdown=remaining,
        )
        display_image(img)
        timings["render"] = time.perf_counter() - t0
    else:
        dashboard_speed = current_speed if state != "IDLE" else manual_speed_mps
        dashboard = render_dashboard(
            state=state,
            speed_mps=dashboard_speed,
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
            display_image(dashboard)

        timings["render"] = time.perf_counter() - t0

    timings["loop_total"] = time.perf_counter() - loop_t0
    if frame_counter % PRINT_EVERY_N_FRAMES == 0:
        print_timing(state, timings)

cv2.destroyAllWindows()