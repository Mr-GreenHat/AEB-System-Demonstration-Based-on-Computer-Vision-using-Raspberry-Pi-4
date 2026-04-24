import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from ego_sim import EgoVehicle
from ipm.webcam_distance_test import main

DEBUG_TIMING = True
PRINT_EVERY_N_FRAMES = 1   # set to 30 later if the terminal spam slows everything down

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

ROBOT_Y = WORLD_HEIGHT // 2
OBJECT_Y = WORLD_HEIGHT // 2
BOX_W = 40
BOX_H = 40

MIN_VALID_DISTANCE = 0.10

# Init behavior
INIT_REQUIRED_SAMPLES = 8
INIT_MAX_WAIT_SEC = 2.0

# ============================================================
# Braking / risk logic aligned to the presentation
# Safe:           TTC > 4.6 s
# FCW:            2.9 s < TTC <= 4.6 s
# Partial braking:1.1 s < TTC <= 2.9 s
# Emergency:      TTC <= 1.1 s
# Partial braking is about 30%, full emergency braking is 100%.
# Full emergency deceleration target: 8 m/s^2
# ============================================================
SAFE_TTC = 4.6
FCW_TTC = 2.9
PARTIAL_TTC = 1.1

PARTIAL_BRAKE_LEVEL = 0.30
FULL_BRAKE_LEVEL = 1.00

FULL_BRAKE_DECEL = 8.0
PARTIAL_BRAKE_DECEL = FULL_BRAKE_DECEL * PARTIAL_BRAKE_LEVEL

STOP_SPEED_EPS = 0.05

# Demo speed controls if encoder is not connected yet
USE_MANUAL_SPEED = True
MANUAL_SPEED_STEP = 0.10
MAX_DEMO_SPEED = 5.0

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

def set_warning_output(enabled: bool):
    """
    Replace this with buzzer / LED output for FCW.
    """
    pass

def set_brake_output(level: float):
    """
    Replace this with GPIO / relay / serial output for the real brake actuator.

    level:
        0.0 = no brake
        0.3 = partial brake
        1.0 = full brake
    """
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
    """
    Used ONLY during INIT to lock the starting distance.
    """
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
    if not time_log:
        return

    fcw_idx = first_state_idx(state_log, "FCW")
    partial_idx = first_state_idx(state_log, "PARTIAL")
    emergency_idx = first_state_idx(state_log, "EMERGENCY")
    stop_idx = first_state_idx(state_log, "STOP")
    crash_idx = first_state_idx(state_log, "CRASH")

    # 1) Distance vs Time
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

    # 2) Speed vs Time
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

    # 3) TTC vs Time
    plt.figure(figsize=(9, 4))
    plt.plot(time_log, ttc_log, label="TTC")
    plt.axhline(SAFE_TTC, linestyle="--", label="Safe threshold")
    plt.axhline(FCW_TTC, linestyle="--", label="FCW threshold")
    plt.axhline(PARTIAL_TTC, linestyle=":", label="Emergency threshold")
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

    # 4) Actual vs theoretical stopping comparison
    plt.figure(figsize=(9, 4))
    plt.plot(time_log, travel_log, label="Actual travel (virtual)")
    plt.plot(time_log, stop_req_log, label="Required stopping distance")
    plt.plot(time_log, distance_log, linestyle="--", label="Remaining distance")
    if partial_idx is not None:
        plt.axvline(time_log[partial_idx], linestyle="--", label="Partial brake trigger")
    if emergency_idx is not None:
        plt.axvline(time_log[emergency_idx], linestyle="--", label="Emergency brake trigger")
    plt.xlabel("Time (s)")
    plt.ylabel("Meters")
    plt.title("Stopping Distance Comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# Demo speed fallback
# Replace this with encoder / serial input later.
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

# Brake event freeze values
brake_trigger_speed = 0.0
brake_required_stop_distance = 0.0
brake_trigger_mode = None

# Logs
time_log = []
distance_log = []
speed_log = []
ttc_log = []
travel_log = []
stop_req_log = []
state_log = []
sim_time = 0.0

# Current moving speed of the virtual car
current_speed = 0.0

# Current brake actuation level
brake_level = 0.0

# Current TTC
ttc = math.inf
status = "SAFE"

# ============================================================
# Main loop
# ============================================================
for frame, tracks in main(yield_every_frame=True):
    frame_counter += 1
    loop_t0 = time.perf_counter()
    timings = {}

    # --------------------------------------------------------
    # Timing: dt calculation
    # --------------------------------------------------------
    t0 = time.perf_counter()
    now = time.perf_counter()
    if last_frame_time is None:
        dt = 1.0 / 30.0
    else:
        dt = max(now - last_frame_time, 1e-3)
    last_frame_time = now
    timings["dt"] = time.perf_counter() - t0

    # --------------------------------------------------------
    # Keyboard controls
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Draw camera preview
    # --------------------------------------------------------
    t0 = time.perf_counter()
    cv2.imshow("CV + Tracking", frame)
    timings["camera_imshow"] = time.perf_counter() - t0

    # --------------------------------------------------------
    # Get live distance
    # --------------------------------------------------------
    t0 = time.perf_counter()
    live_distance = get_closest_live_distance(tracks)
    last_live_distance = live_distance if live_distance is not None else last_live_distance
    timings["live_distance"] = time.perf_counter() - t0

    # --------------------------------------------------------
    # IDLE
    # --------------------------------------------------------
    if state == "IDLE":
        t0 = time.perf_counter()

        world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

        cv2.putText(world, "IDLE - press I to initialize", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(world, f"Manual speed: {manual_speed_mps:.2f} m/s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(world, "W/S adjust speed, SPACE zeroes it", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if last_live_distance is not None:
            cv2.putText(world, f"Live camera distance: {last_live_distance:.2f} m", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
        else:
            cv2.putText(world, "Live camera distance: none", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

        cv2.imshow("2D World", world)
        timings["idle_render"] = time.perf_counter() - t0

        timings["loop_total"] = time.perf_counter() - loop_t0
        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_timing("IDLE", timings)
        continue

    # --------------------------------------------------------
    # INIT
    # --------------------------------------------------------
    if state == "INIT":
        t0 = time.perf_counter()

        world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

        if live_distance is not None:
            init_samples.append(live_distance)

        elapsed = now - init_start_time if init_start_time is not None else 0.0

        cv2.putText(world, "INIT - locking initial distance", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(world, f"Samples: {len(init_samples)}/{INIT_REQUIRED_SAMPLES}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(world, f"Elapsed: {elapsed:.2f} s", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        if live_distance is not None:
            cv2.putText(world, f"Current live distance: {live_distance:.2f} m", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
        else:
            cv2.putText(world, "Current live distance: waiting for detection", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

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

        cv2.imshow("2D World", world)
        timings["init_render"] = time.perf_counter() - t0

        timings["loop_total"] = time.perf_counter() - loop_t0
        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_timing("INIT", timings)
        continue

    # --------------------------------------------------------
    # RUN / FCW / PARTIAL / EMERGENCY / STOP / CRASH
    # --------------------------------------------------------
    t0 = time.perf_counter()
    wheel_speed_mps = read_wheel_speed_mps()
    timings["read_speed"] = time.perf_counter() - t0

    # Compute current distance and TTC first
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
            brake_level = PARTIAL_BRAKE_LEVEL
            brake_trigger_speed = current_speed
            brake_trigger_mode = "PARTIAL"
            brake_required_stop_distance = stopping_distance(brake_trigger_speed, PARTIAL_BRAKE_DECEL)
            set_warning_output(True)
            set_brake_output(brake_level)

        else:  # EMERGENCY
            state = "EMERGENCY"
            warning_on = True
            brake_on = True
            brake_level = FULL_BRAKE_LEVEL
            brake_trigger_speed = current_speed
            brake_trigger_mode = "EMERGENCY"
            brake_required_stop_distance = stopping_distance(brake_trigger_speed, FULL_BRAKE_DECEL)
            set_warning_output(True)
            set_brake_output(brake_level)

    elif state == "PARTIAL":
        warning_on = True
        brake_on = True
        brake_level = PARTIAL_BRAKE_LEVEL
        set_warning_output(True)
        set_brake_output(brake_level)

        current_speed = max(current_speed - PARTIAL_BRAKE_DECEL * dt, 0.0)
        ego.set_speed(current_speed)
        robot_z += current_speed * dt

        virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance is not None else 0.0
        ttc = ttc_from(virtual_distance, current_speed)
        status = "PARTIAL"

        if virtual_distance <= 0.0 and current_speed > STOP_SPEED_EPS:
            state = "CRASH"
            current_speed = 0.0
            ego.set_speed(0.0)
            virtual_distance = 0.0
            brake_level = FULL_BRAKE_LEVEL
            set_brake_output(brake_level)

        elif current_speed <= STOP_SPEED_EPS:
            state = "STOP"
            current_speed = 0.0
            ego.set_speed(0.0)
            brake_on = True
            brake_level = PARTIAL_BRAKE_LEVEL
            set_brake_output(brake_level)

    elif state == "EMERGENCY":
        warning_on = True
        brake_on = True
        brake_level = FULL_BRAKE_LEVEL
        set_warning_output(True)
        set_brake_output(brake_level)

        current_speed = max(current_speed - FULL_BRAKE_DECEL * dt, 0.0)
        ego.set_speed(current_speed)
        robot_z += current_speed * dt

        virtual_distance = max(locked_initial_distance - robot_z, 0.0) if locked_initial_distance is not None else 0.0
        ttc = ttc_from(virtual_distance, current_speed)
        status = "EMERGENCY"

        if virtual_distance <= 0.0 and current_speed > STOP_SPEED_EPS:
            state = "CRASH"
            current_speed = 0.0
            ego.set_speed(0.0)
            virtual_distance = 0.0
            brake_level = FULL_BRAKE_LEVEL
            set_brake_output(brake_level)

        elif current_speed <= STOP_SPEED_EPS:
            state = "STOP"
            current_speed = 0.0
            ego.set_speed(0.0)
            brake_on = True
            brake_level = FULL_BRAKE_LEVEL
            set_brake_output(brake_level)

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
        brake_level = FULL_BRAKE_LEVEL
        set_warning_output(True)
        set_brake_output(brake_level)
        virtual_distance = 0.0
        ttc = math.inf
        status = "CRASH_RISK"
    timings["logic"] = time.perf_counter() - t0

    # Ensure values exist for display/logging
    t0 = time.perf_counter()
    if state not in ("RUN", "FCW", "PARTIAL", "EMERGENCY"):
        if not math.isfinite(ttc):
            ttc = math.inf
        if virtual_distance is None:
            virtual_distance = 0.0
    timings["sanity"] = time.perf_counter() - t0

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------
    t0 = time.perf_counter()
    sim_time += dt
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

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    t0 = time.perf_counter()
    world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

    cv2.putText(world, f"STATE: {state}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(world, f"Speed: {current_speed:.2f} m/s", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if locked_initial_distance is not None:
        cv2.putText(world, f"Initial distance d0: {locked_initial_distance:.2f} m", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    dist_text = f"{virtual_distance:.2f} m" if virtual_distance is not None else "n/a"
    ttc_text = f"{ttc:.2f} s" if math.isfinite(ttc) else "inf"

    cv2.putText(world, f"Virtual distance: {dist_text}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if state in ("PARTIAL", "EMERGENCY"):
        req_text = f"{brake_required_stop_distance:.2f} m"
    else:
        req_text = "n/a"

    cv2.putText(world, f"TTC: {ttc_text}   Stop dist: {req_text}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(world, f"Brake: {'ON' if brake_on else 'OFF'}   Warning: {'ON' if warning_on else 'OFF'}",
                (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    cv2.putText(world, f"Brake level: {brake_level:.2f}   Mode: {status}",
                (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    span_m = max(locked_initial_distance if locked_initial_distance else 5.0, 5.0)
    display_span_px = WORLD_WIDTH - 140
    left_x = 60

    robot_x = int(left_x + min(max((robot_z / span_m) * display_span_px, 0.0), display_span_px))
    if locked_initial_distance is not None:
        object_x = int(left_x + min(max((locked_initial_distance / span_m) * display_span_px, 0.0), display_span_px))
    else:
        object_x = WORLD_WIDTH - 80

    cv2.rectangle(world,
                  (robot_x, ROBOT_Y - BOX_H // 2),
                  (robot_x + BOX_W, ROBOT_Y + BOX_H // 2),
                  (0, 255, 0) if state != "CRASH" else (0, 0, 255),
                  -1)

    cv2.rectangle(world,
                  (object_x, OBJECT_Y - BOX_H // 2),
                  (object_x + BOX_W, OBJECT_Y + BOX_H // 2),
                  (0, 0, 255),
                  -1)

    cv2.line(world,
             (robot_x + BOX_W, ROBOT_Y),
             (object_x, OBJECT_Y),
             (255, 255, 255),
             2)

    if state == "CRASH":
        cv2.putText(world, "COLLISION OCCURRED", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    elif state == "STOP":
        cv2.putText(world, "STOPPED SAFELY", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    elif state == "PARTIAL":
        cv2.putText(world, "PARTIAL BRAKING", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    elif state == "EMERGENCY":
        cv2.putText(world, "EMERGENCY BRAKING", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    elif state == "FCW":
        cv2.putText(world, "FORWARD COLLISION WARNING", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("2D World", world)
    timings["render"] = time.perf_counter() - t0

    # --------------------------------------------------------
    # Plot once when run ends
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if not plots_shown and state in ("STOP", "CRASH"):
        plots_shown = True
        plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log)
    timings["plot"] = time.perf_counter() - t0

    timings["loop_total"] = time.perf_counter() - loop_t0
    if frame_counter % PRINT_EVERY_N_FRAMES == 0:
        print_timing(state, timings)

cv2.destroyAllWindows()

# If user quits before a STOP/CRASH, still show what we collected.
if time_log and not plots_shown:
    plot_results(time_log, distance_log, speed_log, ttc_log, travel_log, stop_req_log, state_log)