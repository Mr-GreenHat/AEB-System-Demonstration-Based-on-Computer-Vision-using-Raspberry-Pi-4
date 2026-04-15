import cv2
import numpy as np

from ego_sim import EgoVehicle
from ipm.webcam_distance_test import main
from braking.TTC_calculate import TTCSystem

# -----------------------------
# Config
# -----------------------------
SCALE = 10.0
WORLD_WIDTH = 800
WORLD_HEIGHT = 200

ROBOT_Y = WORLD_HEIGHT // 2
OBJECT_Y = WORLD_HEIGHT // 2

PIXELS_PER_METER = 50

ego = EgoVehicle()
ego.set_speed(0.0)

ttc_system = TTCSystem()

# 🔥 smoothing (important)
prev_closest = None

for frame, tracks in main(yield_every_frame=True):

    cv2.imshow("CV + Tracking", frame)

    robot_z, dt = ego.update()

    world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)

    if not tracks:
        cv2.imshow("2D World", world)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    relative_distances = []

    for tid, tr in tracks.items():

        if tr.smoothed_depth is None or not np.isfinite(tr.smoothed_depth):
            continue

        target_z = tr.smoothed_depth * SCALE

        relative_distance = target_z - robot_z
        relative_distance = max(relative_distance, 0.0)

        relative_distances.append(relative_distance)

        # -----------------------------
        # VISUALIZATION
        # -----------------------------
        robot_x = int(robot_z * PIXELS_PER_METER)
        object_x = int(target_z * PIXELS_PER_METER)

        robot_x = min(robot_x, WORLD_WIDTH - 50)
        object_x = min(object_x, WORLD_WIDTH - 50)

        cv2.rectangle(world, (object_x, OBJECT_Y - 20),
                      (object_x + 40, OBJECT_Y + 20),
                      (0, 0, 255), -1)

        cv2.rectangle(world, (robot_x, ROBOT_Y - 20),
                      (robot_x + 40, ROBOT_Y + 20),
                      (0, 255, 0), -1)

        cv2.line(world,
                 (robot_x + 20, ROBOT_Y),
                 (object_x, OBJECT_Y),
                 (255, 255, 255), 2)

    # -----------------------------
    # SMOOTHING (CRITICAL)
    # -----------------------------
    if relative_distances:
        closest = min(relative_distances)

        if prev_closest is None:
            smoothed = closest
        else:
            smoothed = 0.7 * prev_closest + 0.3 * closest

        prev_closest = smoothed
        distances_for_ttc = [smoothed]
    else:
        distances_for_ttc = []

    # -----------------------------
    # TTC (FIXED)
    # -----------------------------
    result = ttc_system.update(distances_for_ttc, ego.velocity)

    # -----------------------------
    # CONTROL FROM TTC
    # -----------------------------
    if result:
        status = result["status"]

        if status == "DARURAT":
            ego.set_speed(0.0)
        elif status == "PARSIAL":
            ego.set_speed(0.2)
        elif status == "AMAN":
            ego.set_speed(0.8)
        else:
            ego.set_speed(0.0)

        cv2.putText(world,
                    f"TTC: {result['ttc']:.2f}s | {status}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    # -----------------------------
    # DISPLAY DISTANCE
    # -----------------------------
    if relative_distances:
        cv2.putText(world,
                    f"Dist: {prev_closest:.2f} m",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    cv2.imshow("2D World", world)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break