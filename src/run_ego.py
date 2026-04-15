import cv2
import numpy as np
import time
import math

from ego_sim import EgoVehicle
from ipm.webcam_distance_test import main

# -----------------------------
# Config
# -----------------------------
SCALE = 10.0  # keep this for now
WORLD_WIDTH = 800
WORLD_HEIGHT = 200

ROBOT_Y = WORLD_HEIGHT // 2
OBJECT_Y = WORLD_HEIGHT // 2

PIXELS_PER_METER = 50

# Speed limits
MAX_SPEED = 0.8
MIN_SPEED = 0.0

# Braking / acceleration limits (smooth control)
MAX_ACCEL = 0.6     # m/s^2
MAX_DECEL = 1.2     # m/s^2

# Detection handling
MIN_VALID_DISTANCE = 0.1
STALE_TRACK_TIMEOUT = 1.0
REQUIRE_VALID_TRACKS = True

# TTC stability
MIN_CLOSING_SPEED = 1e-3
MAX_CLOSING_SPEED = 10.0
VELOCITY_ALPHA = 0.7
TTC_ALPHA = 0.8

# TTC thresholds for status display
EMERGENCY_TTC = 1.2
PARTIAL_TTC = 3.0
SAFE_TTC = 3.5

# -----------------------------
# Helpers
# -----------------------------
def depth_to_meters(depth_value: float) -> float:
    return max(depth_value * SCALE, 0.0)

def ttc_to_target_speed(ttc: float) -> float:
    """
    Continuous speed curve instead of hard state jumps.
    Lower TTC => lower target speed.
    """
    if not math.isfinite(ttc):
        return 0.0

    if ttc <= 0.8:
        return 0.0
    elif ttc <= 1.2:
        return 0.05
    elif ttc <= 1.8:
        return 0.12
    elif ttc <= 2.5:
        return 0.25
    elif ttc <= 3.5:
        return 0.45
    else:
        return MAX_SPEED

def ttc_status(ttc: float) -> str:
    if not math.isfinite(ttc):
        return "STOP"
    if ttc <= EMERGENCY_TTC:
        return "DARURAT"
    if ttc <= PARTIAL_TTC:
        return "PARSIAL"
    return "AMAN"

def ramp_speed(current_speed: float, target_speed: float, dt: float) -> float:
    """
    Limit speed change per frame to avoid sudden jumps.
    """
    if target_speed > current_speed:
        return min(current_speed + MAX_ACCEL * dt, target_speed)
    else:
        return max(current_speed - MAX_DECEL * dt, target_speed)

def reset_ego(ego: EgoVehicle):
    """
    Try to fully reset the ego vehicle.
    """
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

# -----------------------------
# Init
# -----------------------------
ego = EgoVehicle()
ego.set_speed(0.0)

track_history = {}
last_frame_time = None
robot_z = 0.0

smoothed_ttc = None
current_speed_command = 0.0

# -----------------------------
# Main loop
# -----------------------------
for frame, tracks in main(yield_every_frame=True):
    now = time.perf_counter()

    if last_frame_time is None:
        frame_dt = 1.0 / 30.0
    else:
        frame_dt = max(now - last_frame_time, 1e-3)
    last_frame_time = now

    # Update ego using previously commanded speed
    robot_z, ego_dt = ego.update()

    world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)
    cv2.imshow("CV + Tracking", frame)

    # Remove stale tracks
    stale_ids = []
    for tid, hist in track_history.items():
        if now - hist["last_seen"] > STALE_TRACK_TIMEOUT:
            stale_ids.append(tid)
    for tid in stale_ids:
        del track_history[tid]

    valid_tracks = []
    min_ttc = math.inf
    has_valid_distance = False

    # If nothing detected at all, hard reset
    if not tracks:
        robot_z = 0.0
        smoothed_ttc = None
        current_speed_command = 0.0
        track_history.clear()
        reset_ego(ego)

        cv2.putText(
            world,
            f"No detection | RESET | v={ego.velocity:.2f} m/s",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            world,
            f"Robot z: {robot_z:.2f} m",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("2D World", world)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Process each track
    for tid, tr in tracks.items():
        depth = getattr(tr, "smoothed_depth", None)

        if depth is None:
            continue

        try:
            depth = float(depth)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(depth):
            continue

        target_z = depth_to_meters(depth)
        relative_distance = max(target_z - robot_z, 0.0)

        if relative_distance < MIN_VALID_DISTANCE:
            continue

        has_valid_distance = True

        hist = track_history.setdefault(
            tid,
            {
                "prev_distance": None,
                "velocity": 0.0,
                "last_seen": now,
            },
        )

        # Raw closing speed from frame-to-frame distance change
        if hist["prev_distance"] is None:
            raw_velocity = 0.0
        else:
            raw_velocity = (hist["prev_distance"] - relative_distance) / frame_dt

        # Smooth the closing speed
        smoothed_velocity = (
            VELOCITY_ALPHA * hist["velocity"] + (1.0 - VELOCITY_ALPHA) * raw_velocity
        )

        # Clamp nonsense
        if smoothed_velocity < 0.0:
            smoothed_velocity = 0.0
        if smoothed_velocity > MAX_CLOSING_SPEED:
            smoothed_velocity = MAX_CLOSING_SPEED

        hist["velocity"] = smoothed_velocity
        hist["prev_distance"] = relative_distance
        hist["last_seen"] = now

        # TTC
        if smoothed_velocity > MIN_CLOSING_SPEED:
            raw_ttc = relative_distance / smoothed_velocity
        else:
            raw_ttc = math.inf

        # Smooth TTC too, so the decision does not jump
        if math.isfinite(raw_ttc):
            if smoothed_ttc is None or not math.isfinite(smoothed_ttc):
                smoothed_ttc = raw_ttc
            else:
                smoothed_ttc = TTC_ALPHA * smoothed_ttc + (1.0 - TTC_ALPHA) * raw_ttc
        else:
            smoothed_ttc = math.inf

        valid_tracks.append(
            {
                "tid": tid,
                "distance": relative_distance,
                "target_z": target_z,
                "ttc": smoothed_ttc,
                "closing_speed": smoothed_velocity,
            }
        )

        if smoothed_ttc < min_ttc:
            min_ttc = smoothed_ttc

        # Visualization
        robot_x = int(robot_z * PIXELS_PER_METER)
        object_x = int(target_z * PIXELS_PER_METER)

        robot_x = max(0, min(robot_x, WORLD_WIDTH - 50))
        object_x = max(0, min(object_x, WORLD_WIDTH - 50))

        cv2.rectangle(
            world,
            (object_x, OBJECT_Y - 20),
            (object_x + 40, OBJECT_Y + 20),
            (0, 0, 255),
            -1,
        )

        cv2.rectangle(
            world,
            (robot_x, ROBOT_Y - 20),
            (robot_x + 40, ROBOT_Y + 20),
            (0, 255, 0),
            -1,
        )

        cv2.line(
            world,
            (robot_x + 20, ROBOT_Y),
            (object_x, OBJECT_Y),
            (255, 255, 255),
            2,
        )

        ttc_text = "inf" if not math.isfinite(smoothed_ttc) else f"{smoothed_ttc:.2f}s"
        cv2.putText(
            world,
            f"ID {tid} d={relative_distance:.2f}m TTC={ttc_text}",
            (20, 90 + 20 * (len(valid_tracks) - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    # Control logic
    if REQUIRE_VALID_TRACKS and not has_valid_distance:
        robot_z = 0.0
        current_speed_command = 0.0
        smoothed_ttc = None
        track_history.clear()
        reset_ego(ego)
        state = "STOP"
        target_speed = 0.0
        min_ttc = math.inf
    else:
        state = ttc_status(min_ttc)
        target_speed = ttc_to_target_speed(min_ttc)

        # Smooth actual command to avoid sudden braking or acceleration
        current_speed_command = ramp_speed(current_speed_command, target_speed, frame_dt)
        ego.set_speed(current_speed_command)

    # Display info
    ttc_text = "inf" if not math.isfinite(min_ttc) else f"{min_ttc:.2f}s"

    cv2.putText(
        world,
        f"TTC: {ttc_text} | {state} | cmd_v={current_speed_command:.2f} m/s",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        f"Robot z: {robot_z:.2f} m",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if valid_tracks:
        closest_dist = min(t["distance"] for t in valid_tracks)
        cv2.putText(
            world,
            f"Closest d: {closest_dist:.2f} m",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.imshow("2D World", world)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()