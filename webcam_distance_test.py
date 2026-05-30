# -*- coding: utf-8 -*-
"""
Raspberry Pi 4 High-Performance Edition
YOLOv8n INT8 ONNX Runtime + calibrated IPM distance + IoU Tracking

Fixes:
- Uses IPM raw depth, then calibrates raw depth to real measured distance.
- Calibration uses:
    raw 4  -> real 3 m
    raw 12 -> real 6 m
    raw 53 -> real 9 m
- No double display when yield_every_frame=True.
- No cv2.waitKey() when yield_every_frame=True.
- Label moves below bbox if hidden at top.
"""

import os
import time
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort


# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "ncnn_param": r"/home/adji/ttc_project/yolov8n_ncnn_model/model.ncnn.param",
    "ncnn_bin":   r"/home/adji/ttc_project/yolov8n_ncnn_model/model.ncnn.bin",
    "names": r"/home/adji/ttc_project/coco.names",
}

CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480

YOLO_INPUT_SIZE = 320

DETECTION_INTERVAL = 1
MAX_MISSED_FRAMES = 8
MATCH_IOU_THRESH = 0.30
YOLO_CONF_THRES = 0.35
YOLO_IOU_THRES = 0.45

# COCO: 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck
TARGET_CLASSES = {0, 2, 3, 5, 7}

VERBOSE_DEPTH = False  # set True only for calibration; spams terminal every frame

camera_intrinsics_path = r"/home/adji/ttc_project/calibration/camera_intrinsics640x480.npz"


# -----------------------------
# Load intrinsics
# -----------------------------
data = np.load(camera_intrinsics_path)
mtx = data["K"].astype(np.float64)
dist = data["dist"].astype(np.float64)

UNDISTORT_MAP1, UNDISTORT_MAP2 = cv2.initUndistortRectifyMap(
    mtx,
    dist,
    None,
    mtx,
    (FRAME_W, FRAME_H),
    cv2.CV_16SC2,
)


# -----------------------------
# Calibration config
# -----------------------------
# Use 4 distances for calibration.
# You will fill RAW_CALIB_POINTS after measuring raw IPM medians.
REAL_CALIB_POINTS = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)

RAW_CALIB_POINTS = np.array([
    3.74,   # raw median at real 3 m
    6.38,   # raw median at real 5 m
    12.34,  # raw median at real 7 m
    21.95,  # raw median at real 9 m
], dtype=np.float64)


def calibration_ready():
    return np.all(np.isfinite(RAW_CALIB_POINTS)) and np.all(RAW_CALIB_POINTS > 0)


def calibrate_raw_depth(raw_depth_m):
    """
    Calibrate raw IPM depth using piecewise linear interpolation.
    Outside the calibration range, use linear extrapolation instead of clamping.

    Calibration set:
        raw 3.74  -> real 3 m
        raw 6.38  -> real 5 m
        raw 12.34 -> real 7 m
        raw 21.95 -> real 9 m

    Warning:
    Extrapolation beyond 9 m is less reliable.
    """

    raw = float(raw_depth_m)

    if not np.isfinite(raw) or raw <= 0:
        return np.nan

    if not calibration_ready():
        return raw

    order = np.argsort(RAW_CALIB_POINTS)
    raw_points = RAW_CALIB_POINTS[order]
    real_points = REAL_CALIB_POINTS[order]

    if np.any(np.diff(raw_points) <= 0):
        return np.nan

    # Inside calibration range
    if raw_points[0] <= raw <= raw_points[-1]:
        return float(np.interp(raw, raw_points, real_points))

    # Below first point: extrapolate using first segment
    if raw < raw_points[0]:
        x0, x1 = raw_points[0], raw_points[1]
        y0, y1 = real_points[0], real_points[1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (raw - x0))

    # Above last point: extrapolate using last segment
    if raw > raw_points[-1]:
        x0, x1 = raw_points[-2], raw_points[-1]
        y0, y1 = real_points[-2], real_points[-1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y1 + slope * (raw - x1))


# -----------------------------
# Helpers
# -----------------------------
class _DictObjHolder:
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]


class WebcamVideoStream:
    def __init__(self, src=0, width=640, height=480, fps=60):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            grabbed, frame = self.stream.read()

            if grabbed:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def iou_xyxy(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x4 - x3) * max(0, y4 - y3)

    union = area1 + area2 - inter

    return inter / max(1e-6, union)


def xywh2xyxy(boxes_xywh):
    y = np.zeros_like(boxes_xywh, dtype=np.float32)

    y[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    y[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    y[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    y[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    return y


def clip_boxes_xyxy(boxes, width, height):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

    return boxes


def nms_numpy(boxes_xyxy, scores, iou_thres=0.45):
    if len(boxes_xyxy) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        ious = np.array(
            [iou_xyxy(boxes_xyxy[i], boxes_xyxy[j]) for j in rest],
            dtype=np.float32,
        )

        order = rest[ious <= iou_thres]

    return keep


def letterbox(im, new_shape=(480, 480), color=(114, 114, 114), scaleup=False):
    shape = im.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (
        int(round(shape[1] * r)),
        int(round(shape[0] * r)),
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return im, r, (dw, dh)


# -----------------------------
# IPM
# -----------------------------
class IPM:
    """
    Ground-plane IPM using camera height and pitch.

    Output:
    - X: lateral distance in mm
    - Z: forward ground distance in mm

    The raw Z is then corrected by calibrate_raw_depth().
    """

    def __init__(self, camera_info):
        self.cx = float(camera_info.u_x)
        self.cy = float(camera_info.u_y)
        self.fx = float(camera_info.f_x)
        self.fy = float(camera_info.f_y)

        # Camera height from ground to optical center, in mm
        self.h = float(camera_info.camera_height)

        # Physical downward pitch angle, in degrees
        self.pitch_deg = float(camera_info.pitch)
        self.theta = np.deg2rad(self.pitch_deg)

        print(
            f"[IPM] height={self.h:.1f}mm, "
            f"pitch={self.pitch_deg:.2f}deg"
        )

    def uv2xy(self, uvs):
        u = uvs[0, :].astype(np.float64)
        v = uvs[1, :].astype(np.float64)

        x_n = (u - self.cx) / self.fx
        y_n = (v - self.cy) / self.fy

        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)

        denom = y_n * cos_t + sin_t

        eps = 1e-9
        valid = denom > eps

        Z = np.full_like(y_n, np.nan, dtype=np.float64)
        X = np.full_like(x_n, np.nan, dtype=np.float64)

        Z[valid] = self.h * (cos_t - y_n[valid] * sin_t) / denom[valid]
        X[valid] = x_n[valid] * self.h / denom[valid]

        return np.vstack((X, Z))


# -----------------------------
# YOLOv8 ONNX Detector
# -----------------------------
class YOLOv8ONNX:
    def __init__(
        self,
        model_path,
        class_names,
        conf_thres=YOLO_CONF_THRES,
        iou_thres=YOLO_IOU_THRES,
        input_size=480,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        ishape = self.session.get_inputs()[0].shape

        if len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int):
            self.input_h = int(ishape[2])
            self.input_w = int(ishape[3])
        else:
            self.input_h, self.input_w = self.input_size

    def preprocess(self, frame):
        img, gain, pad = letterbox(
            frame,
            new_shape=(self.input_h, self.input_w),
            scaleup=False,
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        return img, gain, pad

    def scale_boxes(self, boxes, gain, pad, orig_shape):
        boxes = boxes.copy()

        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= gain

        h, w = orig_shape

        return clip_boxes_xyxy(boxes, w, h)

    def _build_detection(self, frame, x1, y1, x2, y2, conf, cls_id):
        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)

        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)

        bw = x2 - x1
        bh = y2 - y1

        center_u = (x1 + x2) * 0.5
        bottom_v = y2

        return {
            "bbox": (int(x1), int(y1), int(bw), int(bh)),
            "center_u": float(center_u),
            "bottom_v": float(bottom_v),
            "world_xy": None,
            "class_name": self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id),
            "class_id": int(cls_id),
            "conf": float(conf),
        }

    def detect(self, frame, target_classes=None):
        blob, gain, pad = self.preprocess(frame)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: blob},
        )

        pred = np.asarray(outputs[0])
        pred = np.squeeze(pred)

        detections = []

        if pred.ndim == 2 and pred.shape[1] == 6:
            boxes = pred[:, :4].astype(np.float32)
            confs = pred[:, 4].astype(np.float32)
            class_ids = pred[:, 5].astype(np.int32)

            keep = confs >= self.conf_thres

            boxes = boxes[keep]
            confs = confs[keep]
            class_ids = class_ids[keep]

            boxes = self.scale_boxes(boxes, gain, pad, frame.shape[:2])

            if target_classes is not None:
                mask = np.array([cid in target_classes for cid in class_ids], dtype=bool)

                boxes = boxes[mask]
                confs = confs[mask]
                class_ids = class_ids[mask]

            for box, conf, cid in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = box.astype(int)

                detections.append(
                    self._build_detection(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        float(conf),
                        int(cid),
                    )
                )

            return detections

        if pred.ndim == 3:
            pred = pred[0]

        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {pred.shape}")

        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (84, 85):
            pred = pred.T

        if pred.shape[1] < 6:
            raise RuntimeError(f"Unexpected prediction shape after transpose: {pred.shape}")

        boxes_xywh = pred[:, :4].astype(np.float32)
        class_scores = pred[:, 4:].astype(np.float32)

        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        confs = class_scores[np.arange(len(class_ids)), class_ids].astype(np.float32)

        keep = confs >= self.conf_thres

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confs = confs[keep]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = xywh2xyxy(boxes_xywh)
        boxes_xyxy = self.scale_boxes(boxes_xyxy, gain, pad, frame.shape[:2])

        if target_classes is not None:
            mask = np.array([cid in target_classes for cid in class_ids], dtype=bool)

            boxes_xyxy = boxes_xyxy[mask]
            class_ids = class_ids[mask]
            confs = confs[mask]

        if len(boxes_xyxy) == 0:
            return []

        for cid in np.unique(class_ids):
            m = class_ids == cid

            cls_boxes = boxes_xyxy[m]
            cls_confs = confs[m]

            keep_idx = nms_numpy(cls_boxes, cls_confs, self.iou_thres)

            for k in keep_idx:
                x1, y1, x2, y2 = cls_boxes[k].astype(int)

                detections.append(
                    self._build_detection(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        float(cls_confs[k]),
                        int(cid),
                    )
                )

        return detections


# -----------------------------
# Track / Tracker
# -----------------------------
class Track:
    def __init__(self, track_id, det, timestamp):
        self.id = track_id
        self.smoothed_depth = np.nan
        self.raw_depth = np.nan
        self.missed_frames = 0
        self.last_update = timestamp

        self.bbox = list(det["bbox"])
        self.center_u = float(det["center_u"])
        self.bottom_v = float(det["bottom_v"])

        self.world_xy = det["world_xy"] if det["world_xy"] is not None else (np.nan, np.nan)
        self.class_name = det["class_name"]

    def get_predicted_bbox(self):
        return tuple(map(int, self.bbox))

    def update(self, det, timestamp):
        # For calibration, use current bbox directly.
        # If you want smoother display later, change this to 0.25.
        alpha = 1.0

        for i in range(4):
            self.bbox[i] = alpha * det["bbox"][i] + (1.0 - alpha) * self.bbox[i]

        self.center_u = alpha * det["center_u"] + (1.0 - alpha) * self.center_u
        self.bottom_v = alpha * det["bottom_v"] + (1.0 - alpha) * self.bottom_v

        if det["world_xy"] is not None:
            self.world_xy = det["world_xy"]

        self.class_name = det["class_name"] or self.class_name
        self.missed_frames = 0
        self.last_update = timestamp

    def recompute_depth(self, ipm):
        try:
            uv = np.array(
                [[self.center_u], [self.bottom_v]],
                dtype=np.float64,
            )

            xy = ipm.uv2xy(uv)

            raw_depth_m = float(xy[1, 0]) / 1000.0
            corrected_depth_m = calibrate_raw_depth(raw_depth_m)

            self.raw_depth = raw_depth_m

            if np.isfinite(corrected_depth_m) and corrected_depth_m > 0:
                # No smoothing during calibration/testing.
                self.smoothed_depth = corrected_depth_m
            else:
                self.smoothed_depth = np.nan

            if VERBOSE_DEPTH:
                print(
                    f"ID={self.id} bottom_v={self.bottom_v:.1f} "
                    f"raw={raw_depth_m:.2f}m corrected={corrected_depth_m:.2f}m"
                )

        except Exception as e:
            print("Depth error:", e)


# -----------------------------
# YOLOv8 NCNN Detector
# -----------------------------
class YOLOv8NCNN:
    def __init__(self, param_path, bin_path, class_names,
                 conf_thres=YOLO_CONF_THRES, iou_thres=YOLO_IOU_THRES, input_size=320):
        import ncnn
        self._ncnn = ncnn
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size

        self.net = ncnn.Net()
        self.net.opt.num_threads = 4
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def detect(self, frame, target_classes=None):
        h_orig, w_orig = frame.shape[:2]
        s = self.input_size

        img, gain, pad = letterbox(frame, new_shape=(s, s), scaleup=False)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mat_in = self._ncnn.Mat.from_pixels(img_rgb, self._ncnn.Mat.PixelType.PIXEL_RGB, s, s)
        mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        _, out = ex.extract("out0")

        out = np.array(out)           # [84, 2100]
        boxes_xywh = out[:4].T        # [2100, 4]
        scores = out[4:].T            # [2100, 80]

        class_ids = scores.argmax(axis=1).astype(np.int32)
        confs = scores[np.arange(len(scores)), class_ids].astype(np.float32)

        keep = confs >= self.conf_thres
        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confs = confs[keep]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = xywh2xyxy(boxes_xywh)
        boxes_xyxy[:, [0, 2]] -= pad[0]
        boxes_xyxy[:, [1, 3]] -= pad[1]
        boxes_xyxy[:, :4] /= gain
        boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, w_orig, h_orig)

        if target_classes is not None:
            mask = np.array([cid in target_classes for cid in class_ids], dtype=bool)
            boxes_xyxy = boxes_xyxy[mask]
            class_ids = class_ids[mask]
            confs = confs[mask]

        if len(boxes_xyxy) == 0:
            return []

        detections = []
        for cid in np.unique(class_ids):
            m = class_ids == cid
            cls_boxes = boxes_xyxy[m]
            cls_confs = confs[m]
            keep_idx = nms_numpy(cls_boxes, cls_confs, self.iou_thres)
            for k in keep_idx:
                x1, y1, x2, y2 = cls_boxes[k].astype(int)
                x1 = max(0, min(w_orig - 1, x1))
                y1 = max(0, min(h_orig - 1, y1))
                x2 = max(0, min(w_orig - 1, x2))
                y2 = max(0, min(h_orig - 1, y2))
                if x2 <= x1: x2 = min(w_orig - 1, x1 + 1)
                if y2 <= y1: y2 = min(h_orig - 1, y1 + 1)
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "center_u": float((x1 + x2) * 0.5),
                    "bottom_v": float(y2),
                    "world_xy": None,
                    "class_name": self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid),
                    "class_id": int(cid),
                    "conf": float(cls_confs[k]),
                })
        return detections


class Tracker:
    def __init__(self, max_missed=MAX_MISSED_FRAMES, match_iou=MATCH_IOU_THRESH):
        self.tracks = {}
        self.next_id = 0
        self.max_missed = max_missed
        self.match_iou = match_iou

    @staticmethod
    def _bbox_xyxy_from_xywh(bbox_xywh):
        x, y, w, h = bbox_xywh
        return (x, y, x + w, y + h)

    def step(self, detections, timestamp):
        assigned = {}
        unmatched_dets = set(range(len(detections)))

        if self.tracks and detections:
            t_ids = list(self.tracks.keys())

            iou_mat = np.array([
                [
                    iou_xyxy(
                        self._bbox_xyxy_from_xywh(
                            self.tracks[tid].get_predicted_bbox()
                        ),
                        self._bbox_xyxy_from_xywh(d["bbox"]),
                    )
                    for d in detections
                ]
                for tid in t_ids
            ], dtype=np.float32)

            while True:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)

                if iou_mat[ti, di] < self.match_iou:
                    break

                assigned[t_ids[ti]] = di
                unmatched_dets.discard(di)

                iou_mat[ti, :] = -1
                iou_mat[:, di] = -1

        for tid, tr in list(self.tracks.items()):
            if tid in assigned:
                tr.update(detections[assigned[tid]], timestamp)
            else:
                tr.missed_frames += 1

        for di in unmatched_dets:
            self.tracks[self.next_id] = Track(
                self.next_id,
                detections[di],
                timestamp,
            )
            self.next_id += 1

        for tid in [
            tid for tid, tr in self.tracks.items()
            if tr.missed_frames > self.max_missed
        ]:
            del self.tracks[tid]

        return self.tracks


# -----------------------------
# Drawing
# -----------------------------
def draw_track(vis, tr):
    x, y, w, h = tr.get_predicted_bbox()

    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    cv2.circle(
        vis,
        (int(tr.center_u), int(tr.bottom_v)),
        5,
        (0, 0, 255),
        -1,
    )

    if np.isfinite(tr.smoothed_depth):
        depth_text = f"{tr.smoothed_depth:.2f}m"
    else:
        depth_text = "na"

    label = f"ID:{tr.id} {tr.class_name} {depth_text}"

    label_x = x
    label_y = y - 10

    if label_y < 25:
        label_y = y + h + 20

    if label_y > vis.shape[0] - 10:
        label_y = vis.shape[0] - 10

    cv2.putText(
        vis,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


# -----------------------------
# Main loop
# -----------------------------
def main(yield_every_frame=False):
    cv2.setNumThreads(0)
    cv2.setUseOptimized(True)

    with open(CONFIG["names"], "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f]

    detector = YOLOv8NCNN(
        CONFIG["ncnn_param"],
        CONFIG["ncnn_bin"],
        class_names=classes,
        conf_thres=YOLO_CONF_THRES,
        iou_thres=YOLO_IOU_THRES,
        input_size=YOLO_INPUT_SIZE,
    )

    cam = _DictObjHolder({
        "f_x": float(mtx[0, 0]),
        "f_y": float(mtx[1, 1]),
        "u_x": float(mtx[0, 2]),
        "u_y": float(mtx[1, 2]),

        # Measure from ground to camera optical center.
        # Unit: mm
        "camera_height": 785.0,

        # Keep your current tested value.
        # Since we now calibrate raw IPM depth, do not keep tuning this blindly.
        "pitch": 0.0,

        "yaw": 0.0,
    })

    ipm = IPM(cam)

    vs = WebcamVideoStream(
        src=CAMERA_INDEX,
        width=FRAME_W,
        height=FRAME_H,
        fps=60,
    ).start()

    tracker = Tracker(
        max_missed=MAX_MISSED_FRAMES,
        match_iou=MATCH_IOU_THRESH,
    )

    frame_count = 0
    fps_ema = 0.0

    try:
        while True:
            t_start = time.perf_counter()

            frame = vs.read()

            if frame is None:
                break

            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            frame = cv2.remap(
                frame,
                UNDISTORT_MAP1,
                UNDISTORT_MAP2,
                interpolation=cv2.INTER_LINEAR,
            )

            frame_count += 1

            detections = []

            if frame_count % DETECTION_INTERVAL == 0:
                detections = detector.detect(
                    frame,
                    target_classes=TARGET_CLASSES,
                )

                for d in detections:
                    uv = np.array(
                        [[d["center_u"]], [d["bottom_v"]]],
                        dtype=np.float64,
                    )

                    xy = ipm.uv2xy(uv)
                    d["world_xy"] = tuple(float(v) for v in xy.reshape(-1))

            tracks = tracker.step(detections, time.time())

            vis = frame.copy()

            for _, tr in tracks.items():
                tr.recompute_depth(ipm)
                draw_track(vis, tr)

            dt = time.perf_counter() - t_start

            if dt > 0:
                cur_fps = 1.0 / dt
                fps_ema = cur_fps if fps_ema == 0 else (0.9 * fps_ema + 0.1 * cur_fps)

            if not yield_every_frame:
                cv2.putText(
                    vis,
                    f"FPS: {fps_ema:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    vis,
                    f"Pitch: {cam.pitch:.1f} deg",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

            if yield_every_frame:
                yield vis, tracks
            else:
                cv2.imshow("Pi 4 YOLOv8 INT8 ONNX + calibrated IPM", vis)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        vs.stop()

        if not yield_every_frame:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    for _ in main(False):
        pass