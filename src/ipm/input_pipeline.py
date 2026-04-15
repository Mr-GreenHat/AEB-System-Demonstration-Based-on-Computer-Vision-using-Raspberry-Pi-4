# -*- coding: utf-8 -*-
import os
import sys
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# IPM class (modified: stores xmin/xmax/ymin/ymax and adds world2bev)
# -----------------------------
class IPM(object):
    """
    Inverse perspective mapping to a bird-eye view. Assume pin-hole camera model.
    `_c` for camera coordinates, `_w` for world coordinates, `uv` for image coords.
    """
    def __init__(self, camera_info, ipm_info):
        self.camera_info = camera_info
        self.ipm_info = ipm_info

        # Translation
        self.T = np.eye(4)
        self.T[2, 3] = -camera_info.camera_height

        # Rotation: yaw then pitch (degrees -> radians)
        _cy = np.cos(camera_info.yaw   * np.pi / 180.)
        _sy = np.sin(camera_info.yaw   * np.pi / 180.)
        _cp = np.cos(camera_info.pitch * np.pi / 180.)
        _sp = np.sin(camera_info.pitch * np.pi / 180.)

        tyaw = np.array([[_cy, 0, -_sy],
                         [0, 1, 0],
                         [_sy, 0, _cy]])
        tyaw_inv = np.array([[_cy, 0, _sy],
                             [0, 1, 0],
                             [-_sy, 0, _cy]])
        tpitch = np.array([[1, 0, 0],
                           [0, _cp, -_sp],
                           [0, _sp, _cp]])
        tpitch_inv = np.array([[1, 0, 0],
                               [0, _cp, _sp],
                               [0, -_sp, _cp]])
        self.R = np.dot(tyaw, tpitch)
        self.R_inv = np.dot(tpitch_inv, tyaw_inv)

        # Intrinsics
        self.K = np.array([[camera_info.f_x, 0, camera_info.u_x],
                           [0, camera_info.f_y, camera_info.u_y],
                           [0, 0, 1]]).astype(float)

        # Ground plane in camera coords: normal_c^T * X = const_c
        self.normal_c = np.dot(self.R, np.array([0,0,1])[:, None])
        self.const_c = np.dot(self.normal_c.T,
                              np.dot(self.R,
                                     np.dot(self.T, np.array([0,0,0,1])[:, None])[:3]))

        # Vanishing point (where forward world direction projects)
        lane_vec_homo_uv = np.dot(self.K, np.dot(self.R, np.array([0,1,0])[:, None]))
        vp = self.vp = lane_vec_homo_uv[:2] / lane_vec_homo_uv[2]

        # Top clamp: ensure ipm area below vanishing point (safety)
        ipm_top = self.ipm_top = ipm_info.top
        uv_limits = self.uv_limits = np.array([[ipm_info.left, ipm_top],
                                               [ipm_info.right, ipm_top],
                                               [vp[0, 0], ipm_top],
                                               [vp[0, 0], ipm_info.bottom]]).T

        # Convert uv_limits -> world xy limits and create grid
        self.xy_limits = self.uv2xy(uv_limits)
        xmin, xmax = min(self.xy_limits[0]), max(self.xy_limits[0])
        ymin, ymax = min(self.xy_limits[1]), max(self.xy_limits[1])

        # store ranges & step for world<->BEV conversions (units: mm)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.stepx = (xmax - xmin) / ipm_info.out_width
        self.stepy = (ymax - ymin) / ipm_info.out_height

        # Print scale info
        print("X range (mm):", xmin, xmax)
        print("Y range (mm):", ymin, ymax)
        print("Width (cm):", (xmax - xmin) / 10)
        print("Height (cm):", (ymax - ymin) / 10)
        print("cm per pixel X:", (xmax - xmin) / 10 / ipm_info.out_width)
        print("cm per pixel Y:", (ymax - ymin) / 10 / ipm_info.out_height)

        # xy_grid: world coords for each BEV pixel (centered)
        self.xy_grid = np.array([[(xmin + self.stepx * (0.5 + j), ymax - self.stepy * (0.5 + i))
                                   for j in range(ipm_info.out_width)]
                                  for i in range(ipm_info.out_height)]).reshape(-1, 2).T

        # uv_grid: corresponding image pixels for each BEV pixel
        self.uv_grid = self.xy2uv(self.xy_grid)
        # mask out-of-range points (keep float for remap interpolation later)
        mask = (self.uv_grid[0] > ipm_info.left) & (self.uv_grid[0] < ipm_info.right) & \
               (self.uv_grid[1] > ipm_top) & (self.uv_grid[1] < ipm_info.bottom)

        # reshape to (h,w) float maps for remap
        uv_reshaped = self.uv_grid.reshape(2, ipm_info.out_height, ipm_info.out_width)
        # convert to float32 for remap
        u_map = uv_reshaped[0].astype(np.float32)
        v_map = uv_reshaped[1].astype(np.float32)

        # replace NaN with -1
        u_map[np.isnan(u_map)] = -1
        v_map[np.isnan(v_map)] = -1

        # keep maps (some entries may be outside; we'll mask them when remapping)
        self.remap_u = u_map
        self.remap_v = v_map

        # Also keep integer masked uv_grid for direct indexing use (if wanted)
        uv_int = np.vstack((self.uv_grid[0].astype(int), self.uv_grid[1].astype(int)))
        uv_int = uv_int * mask.astype(int)
        self.uv_grid_int = tuple(uv_int.reshape(2, ipm_info.out_height, ipm_info.out_width))

    def xy2uv(self, xys):
        # world (x,y) with ground z=0 -> camera coordinates -> project to uv
        xyzs = np.vstack((xys, -self.camera_info.camera_height * np.ones(xys.shape[1])))
        xyzs_c = np.dot(self.K, np.dot(self.R, xyzs))
        return xyzs_c[:2] / xyzs_c[2]

    def uv2xy(self, uvs):
        # given image uv (2,N) -> find intersection with ground z=0, return world (x,y) in mm
        uvs = (uvs - np.array([self.camera_info.u_x, self.camera_info.u_y])[:, None]) / \
              np.array([self.camera_info.f_x, self.camera_info.f_y])[:, None]
        uvs = np.vstack((uvs, np.ones(uvs.shape[1])))
        # find point on camera ray that satisfies plane equation
        xyz_c = (self.const_c / np.dot(self.normal_c.T, uvs)) * uvs
        xy_w = np.dot(self.R_inv, xyz_c)[:2, :]
        return xy_w

    def world2bev(self, xy_w):
        """
        Convert world coordinates (x,y) in mm (shape (2,N)) to BEV pixel coords (col, row) integers.
        col j = (x - xmin) / stepx
        row i = (ymax - y) / stepy
        Returns two arrays (cols, rows) clipped to valid BEV indices.
        """
        if xy_w.ndim == 1:
            xy_w = xy_w.reshape(2, -1)
        xs = xy_w[0]  # mm
        ys = xy_w[1]  # mm
        # compute float indices
        j = (xs - self.xmin) / self.stepx
        i = (self.ymax - ys) / self.stepy
        # round to nearest pixel
        j_int = np.round(j).astype(int)
        i_int = np.round(i).astype(int)
        # clamp
        j_int = np.clip(j_int, 0, self.ipm_info.out_width - 1)
        i_int = np.clip(i_int, 0, self.ipm_info.out_height - 1)
        return j_int, i_int

    def __call__(self, img_gray):
        return self.ipm(img_gray)

    def ipm(self, img_gray):
        # Use cv2.remap with bilinear interpolation to reduce black holes.
        map_x = self.remap_u
        map_y = self.remap_v
        # remap expects maps in float32 and images in uint8
        out_img = cv2.remap(img_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out_img

    def reverse_ipm(self, img_bev, shape=None):
        # Not used in this combined script, keep for completeness
        if shape is None:
            shape = img_bev.shape
        out_img = np.zeros(shape)
        out_img[self.uv_grid_int] = img_bev
        return out_img

# -----------------------------
# Helper class for param dicts
# -----------------------------
class _DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct
    def __getattr__(self, name):
        return self.dct[name]

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    # ---------- CONFIG: update these paths if needed ----------
    CONFIG = {
        "image_path": r"C:\Users\adjip\Documents\python\ASO-IPM\src\ipm\test_img.png",
        "weights": r"src\models\yolov3-tiny.weights",
        "cfg":     r"src\config\yolov3-tiny.cfg",
        "names":   r"src\coco.names",
        "save_detection": "detection_out.png",
        "save_ipm": "ipm_out.png",
        "save_csv": "bev_coordinates.csv"
    }

    # ---------- Camera + IPM params ----------
    camera_info = _DictObjHolder({
        "f_x": 309/5*8,         # focal length x
        "f_y": 344/5*8,         # focal length y
        "u_x": 320,             # optical center x
        "u_y": 240,             # optical center y
        "camera_height": 1000,  # camera height in mm
        "pitch":94,            # rotation degree around x
        "yaw": 0                # rotation degree around y
    })
    ipm_info = _DictObjHolder({
        "input_width": 640,
        "input_height": 480,
        "out_width": 640,
        "out_height": 480,
        "left": 40,
        "right": 600,
        "top": 250,
        "bottom": 400
    })

    # ---------- sanity checks for files ----------
    for p in [CONFIG["image_path"], CONFIG["weights"], CONFIG["cfg"], CONFIG["names"]]:
        if not os.path.exists(p):
            print(f"[ERROR] Required file not found: {p}")
    # (We do not abort here so user sees printed errors; adjust as you want.)

    # ---------- load image ----------
    img_color = cv2.imread(CONFIG["image_path"])
    if img_color is None:
        raise FileNotFoundError(f"Image not found: {CONFIG['image_path']}")
    img_color = cv2.resize(img_color, (ipm_info.input_width, ipm_info.input_height))
    # keep grayscale for IPM
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ---------- create IPM ----------
    ipm = IPM(camera_info, ipm_info)

    # ---------- load class names ----------
    if not os.path.exists(CONFIG["names"]):
        print("[WARN] coco.names file missing, using placeholder class names.")
        classes = ["obj"]
    else:
        with open(CONFIG["names"], "r", encoding="utf-8", errors="ignore") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

    # ---------- load YOLO network ----------
    if not os.path.exists(CONFIG["weights"]) or not os.path.exists(CONFIG["cfg"]):
        print("[WARN] YOLO model files not found. Detection will be skipped.")
        net = None
    else:
        net = cv2.dnn.readNet(CONFIG["weights"], CONFIG["cfg"])

    # ---------- run detection (if net loaded) ----------
    boxes = []
    confidences = []
    class_ids = []
    width = ipm_info.input_width
    height = ipm_info.input_height

    if net is not None:
        # create blob: use swapRB=True (OpenCV loads BGR, YOLO expects RGB)
        blob = cv2.dnn.blobFromImage(img_color, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)

        # handle getUnconnectedOutLayers variety of return shapes
        layer_names = net.getLayerNames()
        out_layer_idxs = net.getUnconnectedOutLayers()
        try:
            out_layer_idxs = out_layer_idxs.flatten().tolist()
        except:
            out_layer_idxs = [int(x) for x in out_layer_idxs]

        output_layers = [layer_names[i - 1] for i in out_layer_idxs]

        # forward
        outs = net.forward(output_layers)

        # parse outputs
        for out in outs:
            for detection in out:
                scores = detection[5:]
                if scores.size == 0:
                    continue
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence > 0.5:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

    # ---------- non-max suppression ----------
    indexes = []
    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        # NMSBoxes returns a list/array of indices. Normalize to a flat python list
        try:
            indexes = indexes.flatten().tolist()
        except:
            # sometimes already a list of lists
            indexes = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indexes]

    # ---------- draw detections and collect bottom points ----------
    img_vis = img_color.copy()
    ground_points = []

    for i, b in enumerate(boxes):
        if i in indexes:
            x, y, w, h = b
            # clamp coords inside image
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width - 1, x + w)
            y1 = min(height - 1, y + h)

            # draw bbox and bottom line in detection image
            cv2.rectangle(img_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.line(img_vis, (x0, y1), (x1, y1), (0, 0, 255), 2)

            label = classes[class_ids[i]] if class_ids and class_ids[i] < len(classes) else "obj"
            cv2.putText(img_vis, f"{label}:{confidences[i]:.2f}", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # add bottom corners as (u,v) pixel coordinates
            ground_points.append([x0, y1])
            ground_points.append([x1, y1])

    # ---------- convert bottom line points (image uv) to world xy ----------
    bev_rows = []
    if len(ground_points) > 0:
        uv = np.array(ground_points).T.astype(float)  # shape (2, N)
        xy_world = ipm.uv2xy(uv)  # shape (2, N) in mm
        print("\nGround points in world coordinates (mm):")
        print(xy_world)  # shape (2, N)
        # prepare CSV
        csv_path = CONFIG["save_csv"]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "object_idx",
                "pixel_u_left","pixel_u_right","pixel_v",
                "world_left_x_mm","world_left_y_mm",
                "world_right_x_mm","world_right_y_mm",
                "depth_mm",
                "distance_mm",
                "bev_left_col","bev_left_row",
                "bev_right_col","bev_right_row",
                "depth_meters",
                "distance_meters"
            ])
            # xy_world ordering matches ground_points: left1, right1, left2, right2, ...
            N = xy_world.shape[1] // 2
            # iterate in pairs
            for k in range(N):
                idx_left = 2 * k
                idx_right = 2 * k + 1
                u_left = ground_points[idx_left][0]
                v = ground_points[idx_left][1]
                u_right = ground_points[idx_right][0]
                # world coords
                left_world = xy_world[:, idx_left]  # [x_mm, y_mm]
                right_world = xy_world[:, idx_right]
                # midpoint of bottom line
                mid_x = (left_world[0] + right_world[0]) / 2.0
                mid_y = (left_world[1] + right_world[1]) / 2.0
                depth_mm = float(mid_y)
                distance_mm = float(np.sqrt(mid_x**2 + mid_y**2))
                depth_meters = depth_mm / 1000.0
                distance_meters = distance_mm / 1000.0
                # BEV pixel coords
                left_col, left_row = ipm.world2bev(left_world)
                right_col, right_row = ipm.world2bev(right_world)
                # write
                writer.writerow([
                    k,
                    int(u_left), int(u_right), int(v),
                    float(left_world[0]), float(left_world[1]),
                    float(right_world[0]), float(right_world[1]),
                    depth_meters,
                    distance_meters,
                    int(left_col[0]), int(left_row[0]),
                    int(right_col[0]), int(right_row[0])
                ])
        print(f"\nSaved BEV coordinates CSV -> {csv_path}")
    else:
        print("\nNo detections -> no ground points to project.")

    # ---------- generate IPM image using the grayscale img ----------
    out_ipm_gray = ipm(img_gray)

    # convert ipm to color (BGR) to draw colored lines
    out_ipm_color = cv2.cvtColor(out_ipm_gray, cv2.COLOR_GRAY2BGR)

    # draw the bottom edges on the IPM image using world->BEV mapping
    if len(ground_points) > 0:
        # xy_world pairs: draw each corresponding line
        num_pairs = xy_world.shape[1] // 2
        for k in range(num_pairs):
            l_idx = 2 * k
            r_idx = 2 * k + 1
            left_w = xy_world[:, l_idx]
            right_w = xy_world[:, r_idx]
            left_col, left_row = ipm.world2bev(left_w)
            right_col, right_row = ipm.world2bev(right_w)
            # extract ints (function returns arrays)
            lc, lr = int(left_col[0]), int(left_row[0])
            rc, rr = int(right_col[0]), int(right_row[0])
            # draw line (red)
            cv2.line(out_ipm_color, (lc, lr), (rc, rr), (0, 0, 255), 2)

            # Overlay depth text at the midpoint of the line
            mid_col, mid_row = (lc + rc) // 2, (lr + rr) // 2
            depth_meters = (left_w[1] + right_w[1]) / 2000.0  # Convert mm to meters and average
            cv2.putText(out_ipm_color, f"{depth_meters:.1f}m", (mid_col, mid_row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ---------- save & show ----------
    cv2.imwrite(CONFIG["save_detection"], img_vis)
    cv2.imwrite(CONFIG["save_ipm"], out_ipm_color)

    print(f"\nSaved detection image -> {CONFIG['save_detection']}")
    print(f"Saved ipm image       -> {CONFIG['save_ipm']}")

    # show windows (will block until key pressed)
    cv2.imshow("detections", img_vis)
    cv2.imshow("ipm", out_ipm_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()