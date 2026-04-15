# full_midas_yolo_no_resize_no_newK.py
# -*- coding: utf-8 -*-
"""
YOLOv3-tiny + MiDaS_small integration, NO resizing, NO getOptimalNewCameraMatrix.
Everything runs at the calibration / camera resolution (FRAME_W x FRAME_H).
Undistort uses the original K (mtx) and dist exactly as requested.

Controls while running:
 - 'q' : quit
 - 'c' : capture current detections' median relative depths and input true distance (meters)
 - 'f' : fit quadratic mapping (rel -> meters)
 - 's' : save coeffs (depth_to_meters_coeffs.npy)
 - 'l' : load coeffs
 - 'd' : toggle display between relative and absolute (if coeffs exist)
"""
import os
import time
import numpy as np
import cv2

# ---------------- CONFIG ----------------
CALIBRATION_FILE = "depth_to_meters_coeffs.npy"
DEPTH_INTERVAL = 3         # run MiDaS every N frames (increase to save CPU)
MIDAS_DEVICE = "cpu"      # "cpu" or "cuda"
FRAME_W = 1920            # you insisted: run at calibration resolution
FRAME_H = 1080
YOLO_INPUT_SIZE = (416, 416)

# YOLO files (update if necessary)
CONFIG = {
    "weights": r"src\models\yolov3-tiny.weights",
    "cfg":     r"src\config\yolov3-tiny.cfg",
    "names":   r"src\coco.names",
    "save_csv": None
}

# Path to camera intrinsics (npz with K and dist). Must exist.
CAM_INTRINSICS = r'C:\Users\adjip\Documents\python\ASO-IPM\calibration\camera_intrinsics.npz'
# ----------------------------------------

if not os.path.exists(CAM_INTRINSICS):
    raise FileNotFoundError(f"Camera intrinsics file not found: {CAM_INTRINSICS}")

data = np.load(CAM_INTRINSICS)
mtx = data['K'].astype(float)   # original K (used directly for undistort)
dist = data['dist'].astype(float)

# ---------------- utilities ----------------
def load_yolo(cfg_path, weights_path):
    if os.path.exists(cfg_path) and os.path.exists(weights_path):
        net = cv2.dnn.readNet(weights_path, cfg_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    else:
        print("[WARN] YOLO files not found - detection disabled.")
        return None

def visualize_depth_map(depth_map):
    """Return a colored visualization of depth_map (no resizing)."""
    if depth_map is None:
        return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    # normalize to 0..255 and invert so nearer appears brighter
    valid = np.isfinite(depth_map)
    if not np.any(valid):
        return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    m = np.nanmin(depth_map[valid])
    M = np.nanmax(depth_map[valid])
    if M - m < 1e-6:
        norm = np.zeros_like(depth_map, dtype=np.float32)
    else:
        norm = (depth_map - m) / (M - m)
    norm8 = (255.0 * (1.0 - np.clip(norm, 0.0, 1.0))).astype(np.uint8)
    colored = cv2.applyColorMap(norm8, cv2.COLORMAP_MAGMA)
    return colored

# ---------------- main ----------------
if __name__ == "__main__":
    # load classes
    if not os.path.exists(CONFIG["names"]):
        classes = ["obj"]
    else:
        with open(CONFIG["names"], "r", encoding="utf-8", errors="ignore") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

    # load YOLO
    net = load_yolo(CONFIG["cfg"], CONFIG["weights"])

    # open camera
    cap = cv2.VideoCapture(1)  # change index if needed
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Change index if necessary.")
    # enforce capture resolution at calibration size (user requested no resize/newK)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # MiDaS setup (try/except; script runs even if torch not present)
    use_midas = True
    midas = None
    midas_transform = None
    torch_device = None
    try:
        import torch
        device = torch.device(MIDAS_DEVICE if (torch.cuda.is_available() and MIDAS_DEVICE == "cuda") else "cpu")
        print("[INFO] Loading MiDaS (small) on device:", device)
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = midas_transforms.small_transform
        torch_device = device
        print("[INFO] MiDaS loaded.")
    except Exception as e:
        print("[WARN] Could not load MiDaS (torch). Depth disabled. Error:", e)
        use_midas = False

    # calibration arrays and coeffs
    calib_rel = []
    calib_abs = []
    depth_to_meters_coeffs = None
    show_abs = True

    frame_idx = 0
    depth_map = None
    last_depth_time = 0.0

    print("[INFO] Running. Controls: c=collect sample, f=fit, s=save coeff, l=load coeff, d=toggle display, q=quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame.")
            break

        # ensure camera actually returned expected resolution; user demanded NO resize -> abort if mismatch
        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            print(f"[ERROR] Camera frame shape {frame.shape[1]}x{frame.shape[0]} != expected {FRAME_W}x{FRAME_H}.")
            print("Because you demanded no resize/no new_K, I will skip this frame. Fix camera or adjust FRAME_W/FRAME_H.")
            # small sleep to avoid tight loop
            time.sleep(0.1)
            continue

        # Undistort using original K and original dist (as you requested)
        # We do NOT call getOptimalNewCameraMatrix or crop/resize. This keeps original FOV and K mapping.
        frame_undist = cv2.undistort(frame, mtx, dist)  # original K used implicitly

        img_vis = frame_undist.copy()

        # MiDaS inference every N frames (no resizing, prediction interpolated to frame size)
        if use_midas and (frame_idx % DEPTH_INTERVAL == 0):
            t0 = time.time()
            img_rgb = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2RGB)
            input_batch = midas_transform(img_rgb).to(torch_device)
            with __import__("torch").no_grad():
                pred = midas(input_batch)
                pred = __import__("torch").nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(frame_undist.shape[0], frame_undist.shape[1]),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_map = pred.cpu().numpy()
            last_depth_time = time.time() - t0
        frame_idx += 1

        # YOLO detection (blobFromImage accepts any input size; mapping later uses FRAME_W/FRAME_H)
        boxes, confidences, class_ids = [], [], []
        if net is not None:
            blob = cv2.dnn.blobFromImage(frame_undist, 1/255.0, YOLO_INPUT_SIZE, swapRB=True, crop=False)
            net.setInput(blob)
            layer_names = net.getLayerNames()
            out_layer_idxs = net.getUnconnectedOutLayers()
            # normalize indices
            if isinstance(out_layer_idxs, (list, tuple, np.ndarray)):
                out_layer_idxs = np.array(out_layer_idxs).flatten().tolist()
            else:
                out_layer_idxs = [int(out_layer_idxs)]
            # convert 1-based -> 0-based if needed
            if any(x > len(layer_names) - 1 for x in out_layer_idxs):
                out_layer_idxs = [x - 1 for x in out_layer_idxs]
            out_layer_idxs = [i for i in out_layer_idxs if 0 <= i < len(layer_names)]
            output_layers = [layer_names[i] for i in out_layer_idxs]
            outs = net.forward(output_layers) if len(output_layers) > 0 else []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    if scores.size == 0:
                        continue
                    class_id = int(np.argmax(scores))
                    if class_id < 0 or class_id >= len(scores):
                        continue
                    confidence = float(scores[class_id])
                    if confidence > 0.5:
                        # map detection (normalized) back to full frame coordinates (FRAME_W x FRAME_H)
                        cx = int(detection[0] * FRAME_W)
                        cy = int(detection[1] * FRAME_H)
                        w = int(detection[2] * FRAME_W)
                        h = int(detection[3] * FRAME_H)
                        x = int(cx - w / 2)
                        y = int(cy - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(class_id)

        # NMS (normalize returned indices)
        if len(boxes) > 0:
            try:
                nms_idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
                if isinstance(nms_idxs, (list, tuple, np.ndarray)):
                    nms_idxs = np.array(nms_idxs).flatten().tolist()
                else:
                    nms_idxs = [int(nms_idxs)]
            except Exception as e:
                print("[WARN] NMS error:", e)
                nms_idxs = list(range(len(boxes)))
        else:
            nms_idxs = []

        # compute median depth per bbox and draw
        median_depths = []
        for i in nms_idxs:
            if i < 0 or i >= len(boxes):
                continue
            x, y, w, h = boxes[i]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(FRAME_W - 1, x + w)
            y1 = min(FRAME_H - 1, y + h)

            cv2.rectangle(img_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = classes[class_ids[i]] if class_ids and class_ids[i] < len(classes) else "obj"
            cv2.putText(img_vis, f"{label}:{confidences[i]:.2f}", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            median_rel = None
            if depth_map is not None:
                roi = depth_map[y0:y1, x0:x1]
                if roi.size > 0:
                    median_rel = float(np.median(roi))
                    median_depths.append((i, median_rel, (x0, y0, x1, y1)))

            if median_rel is None:
                cv2.putText(img_vis, "depth:-", ((x0 + x1)//2, (y0 + y1)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # display converted depths (or relative) on image
        for (i, median_rel, (x0, y0, x1, y1)) in median_depths:
            cx = (x0 + x1)//2
            cy = (y0 + y1)//2
            if depth_to_meters_coeffs is not None and show_abs:
                r = median_rel
                meters = float(depth_to_meters_coeffs[0]*r*r + depth_to_meters_coeffs[1]*r + depth_to_meters_coeffs[2])
                text = f"{meters:.2f} m"
            else:
                text = f"{median_rel:.3f} (rel)"
            cv2.putText(img_vis, text, (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # show depth map (no resizing)
        depth_vis = visualize_depth_map(depth_map)
        cv2.imshow("Depth Map", depth_vis)
        cv2.imshow("Final", img_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if len(median_depths) == 0:
                print("[INFO] No detections with depth to capture.")
            else:
                print("[INFO] Capturing", len(median_depths), "detections. For each, enter true distance (m).")
                for (i, median_rel, (x0, y0, x1, y1)) in median_depths:
                    print(f"Detection {i} bbox {(x0,y0,x1,y1)} median_rel={median_rel:.4f}")
                    try:
                        s = input("Enter real distance for this object in meters (or empty to skip): ").strip()
                        if s == "":
                            print(" - skipped")
                            continue
                        real_m = float(s)
                        calib_rel.append(median_rel)
                        calib_abs.append(real_m)
                        print(f" - saved sample: rel={median_rel:.4f} -> abs={real_m:.3f} m")
                    except Exception as e:
                        print(" - invalid input, skipped.", e)
        elif key == ord('f'):
            if len(calib_rel) < 3:
                print("[WARN] Need at least 3 samples to fit quadratic. Collected:", len(calib_rel))
            else:
                coeffs = np.polyfit(np.array(calib_rel), np.array(calib_abs), 2)
                depth_to_meters_coeffs = coeffs
                print("[INFO] Fitted coefficients (c2, c1, c0):", depth_to_meters_coeffs)
        elif key == ord('s'):
            if depth_to_meters_coeffs is None:
                print("[WARN] No coeffs to save.")
            else:
                np.save(CALIBRATION_FILE, depth_to_meters_coeffs)
                print("[INFO] Saved coeffs to", CALIBRATION_FILE)
        elif key == ord('l'):
            if os.path.exists(CALIBRATION_FILE):
                depth_to_meters_coeffs = np.load(CALIBRATION_FILE)
                print("[INFO] Loaded coeffs:", depth_to_meters_coeffs)
            else:
                print("[WARN] No saved coeffs file found.")
        elif key == ord('d'):
            show_abs = not show_abs

    cap.release()
    cv2.destroyAllWindows()