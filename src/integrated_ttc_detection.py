# integrated_ttc_detection.py
# -*- coding: utf-8 -*-
"""
Integrated TTC runner + High-Performance HUD.
Fixes:
- Latency measurement using high-precision perf_counter
- Correct tuple unpacking (vis_frame, tracks_dict)
- Throttled system stats for Pi 4 performance
"""
import time
import math
import cv2
import numpy as np

# --- Configuration ---
CONSOLE_PRINT_INTERVAL = 1.0  
FPS_EMA_ALPHA = 0.05
STATS_UPDATE_INTERVAL = 60    # Throttled for Pi 4

from ipm.webcam_distance_test import main as original_main
from braking.TTC_calculate import TTCSystem

try:
    import psutil
    _PSUTIL_AVAILABLE = True
    _PROC = psutil.Process()
except:
    _PSUTIL_AVAILABLE = False

def run():
    ttc_system = TTCSystem()
    print("Running High-Performance Integrated TTC (HUD + Depth)...")

    fps_ema = 0.0
    latency_ema = 0.0
    # Use perf_counter for the very first frame to avoid math errors
    prev_recv_time = time.perf_counter() 
    stats_cache = {"sys_cpu": 0, "proc_mem": 0}
    frame_counter = 0

    try:
        # Generator yields (vis_frame, tracks_dict)
        for data in original_main(yield_every_frame=True):
            # Capture receive time immediately
            recv_time = time.perf_counter() 
            frame_counter += 1

            # --- 1. UNPACKING DATA ---
            # This fixes the "'tuple' object has no attribute 'values'" error
            if isinstance(data, tuple):
                vis_frame, tracks_dict = data
            else:
                vis_frame, tracks_dict = None, data

            # --- 2. FPS CALCULATION ---
            dt_recv = recv_time - prev_recv_time
            prev_recv_time = recv_time
            instant_fps = 1.0 / dt_recv if dt_recv > 0 else 0
            fps_ema = (FPS_EMA_ALPHA * instant_fps) + (1 - FPS_EMA_ALPHA) * fps_ema

            # --- 3. SYSTEM STATS (THROTTLED) ---
            if _PSUTIL_AVAILABLE and frame_counter % STATS_UPDATE_INTERVAL == 0:
                stats_cache["sys_cpu"] = psutil.cpu_percent(interval=None)
                stats_cache["proc_mem"] = _PROC.memory_info().rss / (1024 * 1024)

            # --- 4. DEPTH & TTC PROCESSING ---
            # Use perf_counter to measure internal processing cost
            processing_start = time.perf_counter() 
            
            # Extract valid tracks and depths
            active_tracks = [t for t in tracks_dict.values() 
                            if hasattr(t, "smoothed_depth") and np.isfinite(t.smoothed_depth)]
            
            depths = [t.smoothed_depth for t in active_tracks]
            closest_depth = min(depths) if depths else float('inf')
            
            # Update TTC logic
            result = ttc_system.update(depths, time.time()) if depths else None
            
            # Measure latency in milliseconds
            current_lat = (time.perf_counter() - processing_start) * 1000
            latency_ema = (FPS_EMA_ALPHA * current_lat) + (1 - FPS_EMA_ALPHA) * latency_ema

            # --- 5. HUD DRAWING ---
            if vis_frame is not None:
                # Semi-transparent HUD background
                # Note: rectangle with -1 thickness is much faster than alpha blending
                cv2.rectangle(vis_frame, (5, 5), (320, 145), (15, 15, 15), -1) 
                cv2.rectangle(vis_frame, (5, 5), (320, 145), (0, 255, 0), 1)

                # Row 1: Performance (FPS + Precision Latency)
                cv2.putText(vis_frame, f"FPS: {fps_ema:.1f} | Lat: {latency_ema:.2f}ms", 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Row 2: Resources
                cv2.putText(vis_frame, f"CPU: {stats_cache['sys_cpu']}% | RAM: {stats_cache['proc_mem']:.0f}MB", 
                            (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # Row 3: Closest Object Distance
                dist_str = f"{closest_depth:.2f}m" if closest_depth != float('inf') else "N/A"
                cv2.putText(vis_frame, f"CLOSEST OBJ: {dist_str}", 
                            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Row 4: TTC Status (Alert)
                if result:
                    color = (0, 0, 255) if result['status'] == "DARURAT" else (0, 255, 255)
                    cv2.putText(vis_frame, f"TTC: {result['ttc']:.2f}s | {result['status']}", 
                                (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                # --- INDIVIDUAL OVERLAYS ---
                for tr in active_tracks:
                    x, y, w, h = tr.get_predicted_bbox()
                    # Depth Tag near the object
                    tag = f"{tr.smoothed_depth:.1f}m"
                    cv2.putText(vis_frame, tag, (x, y + h + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Use the existing window name from your main script
                cv2.imshow("Pi 4 Optimized AEB System", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()