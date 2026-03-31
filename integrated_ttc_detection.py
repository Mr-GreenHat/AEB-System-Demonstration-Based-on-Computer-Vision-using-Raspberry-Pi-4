# integrated_ttc_detection.py
# -*- coding: utf-8 -*-
"""
Integrated TTC runner + High-Performance HUD.

Keeps the existing detection pipeline and overlays:
- FPS
- latency
- CPU/RAM stats
- closest depth
- TTC status
"""

import time
import cv2
import numpy as np

# --- Configuration ---
CONSOLE_PRINT_INTERVAL = 1.0
FPS_EMA_ALPHA = 0.05
STATS_UPDATE_INTERVAL = 60  # Throttled for Pi 4


def _import_original_main():
    """
    Try the package import first, then fall back to local import.
    This makes the script work whether webcam_distance_test.py lives in:
      - ipm/webcam_distance_test.py
      - ./webcam_distance_test.py
    """
    try:
        from ipm.webcam_distance_test import main as original_main
        return original_main
    except Exception:
        pass

    try:
        from webcam_distance_test import main as original_main
        return original_main
    except Exception as e:
        raise ImportError(
            "Could not import main() from ipm.webcam_distance_test or webcam_distance_test. "
            "Check your project structure."
        ) from e


try:
    from TTC_calculate import TTCSystem
except Exception as e:
    raise ImportError(
        "Could not import TTCSystem from braking.TTC_calculate. "
        "Check that the module exists and is on PYTHONPATH."
    ) from e


try:
    import psutil
    _PSUTIL_AVAILABLE = True
    _PROC = psutil.Process()
except Exception:
    _PSUTIL_AVAILABLE = False
    _PROC = None


def run():
    original_main = _import_original_main()
    ttc_system = TTCSystem()
    print("Running High-Performance Integrated TTC (HUD + Depth)...")

    fps_ema = 0.0
    latency_ema = 0.0
    prev_recv_time = time.perf_counter()
    stats_cache = {"sys_cpu": 0, "proc_mem": 0}
    frame_counter = 0

    try:
        # original_main(yield_every_frame=True) should yield (vis_frame, tracks_dict)
        for data in original_main(yield_every_frame=True):
            recv_time = time.perf_counter()
            frame_counter += 1

            # --- Unpack data safely ---
            if isinstance(data, tuple) and len(data) == 2:
                vis_frame, tracks_dict = data
            else:
                # Fallback if upstream code changes
                vis_frame = None
                tracks_dict = data if isinstance(data, dict) else {}

            # --- FPS calculation ---
            dt_recv = recv_time - prev_recv_time
            prev_recv_time = recv_time
            instant_fps = 1.0 / dt_recv if dt_recv > 0 else 0.0
            fps_ema = (FPS_EMA_ALPHA * instant_fps) + (1.0 - FPS_EMA_ALPHA) * fps_ema

            # --- System stats (throttled) ---
            if _PSUTIL_AVAILABLE and frame_counter % STATS_UPDATE_INTERVAL == 0:
                stats_cache["sys_cpu"] = psutil.cpu_percent(interval=None)
                stats_cache["proc_mem"] = _PROC.memory_info().rss / (1024 * 1024)

            # --- Depth & TTC processing ---
            processing_start = time.perf_counter()

            active_tracks = [
                t for t in tracks_dict.values()
                if hasattr(t, "smoothed_depth") and np.isfinite(t.smoothed_depth)
            ]

            depths = [t.smoothed_depth for t in active_tracks]
            closest_depth = min(depths) if depths else float("inf")

            result = ttc_system.update(depths, time.time()) if depths else None

            current_lat = (time.perf_counter() - processing_start) * 1000.0
            latency_ema = (FPS_EMA_ALPHA * current_lat) + (1.0 - FPS_EMA_ALPHA) * latency_ema

            # --- HUD drawing ---
            if vis_frame is not None:
                cv2.rectangle(vis_frame, (5, 5), (340, 150), (15, 15, 15), -1)
                cv2.rectangle(vis_frame, (5, 5), (340, 150), (0, 255, 0), 1)

                cv2.putText(
                    vis_frame,
                    f"FPS: {fps_ema:.1f} | Lat: {latency_ema:.2f}ms",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    vis_frame,
                    f"CPU: {stats_cache['sys_cpu']}% | RAM: {stats_cache['proc_mem']:.0f}MB",
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                dist_str = f"{closest_depth:.2f}m" if closest_depth != float("inf") else "N/A"
                cv2.putText(
                    vis_frame,
                    f"CLOSEST OBJ: {dist_str}",
                    (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if result:
                    color = (0, 0, 255) if result.get("status") == "DARURAT" else (0, 255, 255)
                    cv2.putText(
                        vis_frame,
                        f"TTC: {result.get('ttc', 0.0):.2f}s | {result.get('status', 'N/A')}",
                        (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                # Individual track overlays
                for tr in active_tracks:
                    x, y, w, h = tr.get_predicted_bbox()
                    tag = f"{tr.smoothed_depth:.1f}m"
                    cv2.putText(
                        vis_frame,
                        tag,
                        (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("Pi 4 Optimized AEB System", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error in main loop: {e}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

