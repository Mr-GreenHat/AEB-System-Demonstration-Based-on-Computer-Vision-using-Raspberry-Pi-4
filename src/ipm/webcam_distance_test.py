# -*- coding: utf-8 -*-
"""
Raspberry Pi 4 High-Performance Edition
YOLOv8n INT8 ONNX Runtime + IPM + IoU Tracking + TTC-ready yield support

Changes:
- Uses yolov8n_int8.onnx
- YOLO input size: 320x320
- Full-frame undistortion via cv2.remap
- TTC-friendly yield_every_frame support
"""

import os
import time
import numpy as np
import cv2
import onnxruntime as ort

# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "onnx": r"C:\Users\adjip\Documents\python\ASO-IPM\src\models\yolov8n_int8.onnx",
    "names": r"C:\Users\adjip\Documents\python\ASO-IPM\src\coco.names",
}

CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480
YOLO_INPUT_SIZE = 320

DETECTION_INTERVAL = 3
MAX_MISSED_FRAMES = 8
MATCH_IOU_THRESH = 0.30
YOLO_CONF_THRES = 0.35
YOLO_IOU_THRES = 0.45

TARGET_CLASSES = {0, 2, 3, 5, 7}
BOX_PADDING = 10
MIN_CONTOUR_AREA = 300

camera_intrinsics_path = r"C:\Users\adjip\Documents\python\ASO-IPM\calibration\camera_intrinsics640x480.npz"

# -----------------------------
# Load intrinsics
# -----------------------------

data = np.load(camera_intrinsics_path)
mtx = data["K"].astype(np.float64)
dist = data["dist"].astype(np.float64)

UNDISTORT_MAP1, UNDISTORT_MAP2 = cv2.initUndistortRectifyMap(
    mtx, dist, None, mtx, (FRAME_W, FRAME_H), cv2.CV_16SC2
)

# -----------------------------
# Helpers
# -----------------------------
class _DictObjHolder:
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]

from threading import Thread

class WebcamVideoStream:
    def __init__(self, src=1, width=640, height=480, fps=60):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

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
        ious = np.array([iou_xyxy(boxes_xyxy[i], boxes_xyxy[j]) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_thres]
    return keep


def letterbox(im, new_shape=(320, 320), color=(114, 114, 114), scaleup=False):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def extract_bottom_point_by_contour(frame, x0, y0, x1, y1, min_area=MIN_CONTOUR_AREA):
    h, w = frame.shape[:2]
    x0c = max(0, int(x0))
    y0c = max(0, int(y0))
    x1c = min(w - 1, int(x1))
    y1c = min(h - 1, int(y1))

    if x1c <= x0c or y1c <= y0c:
        return int((x0 + x1) // 2), int(y1), None

    roi = frame[y0c:y1c, x0c:x1c]
    if roi.size == 0:
        return int((x0 + x1) // 2), int(y1), None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return int((x0 + x1) // 2), int(y1), None

    chosen = max(contours, key=cv2.contourArea)
    if cv2.contourArea(chosen) < min_area:
        return int((x0 + x1) // 2), int(y1), None

    pts = chosen.reshape(-1, 2)
    bottom_v = y0c + int(pts[np.argmax(pts[:, 1]), 1])
    M = cv2.moments(chosen)

    if M["m00"] != 0:
        center_u = x0c + int(M["m10"] / M["m00"])
    else:
        center_u = x0c + int(np.mean(pts[:, 0]))

    return center_u, bottom_v, pts + np.array([x0c, y0c])

def undistort_pixel(u, v, mtx, dist):
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    return float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])

# -----------------------------
# IPM
# -----------------------------
class IPM:
    def __init__(self, camera_info):
        self.cx = camera_info.u_x
        self.cy = camera_info.u_y
        self.fx = camera_info.f_x
        self.fy = camera_info.f_y
        self.h = camera_info.camera_height

    def uv2xy(self, uvs):
        u = uvs[0, :].astype(np.float64)
        v = uvs[1, :].astype(np.float64)

        # normalized image coords
        x_n = (u - self.cx) / self.fx
        y_n = (v - self.cy) / self.fy

        # avoid division by zero
        y_n_safe = np.where(np.abs(y_n) < 1e-9, np.nan, y_n)

        # FIX: Remove the negative sign. Ground is at Y = +h in OpenCV coords.
        Z = self.h / y_n_safe
        X = x_n * Z

        return np.vstack((X, Z))

# -----------------------------
# YOLOv8 ONNX Detector
# -----------------------------
class YOLOv8ONNX:
    def __init__(self, model_path, class_names, conf_thres=YOLO_CONF_THRES, iou_thres=YOLO_IOU_THRES, input_size=320):
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
        img, gain, pad = letterbox(frame, new_shape=(self.input_h, self.input_w), scaleup=False)
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

    def detect(self, frame, target_classes=None):
        blob, gain, pad = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

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
                detections.append(self._build_detection(frame, x1, y1, x2, y2, float(conf), int(cid)))

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
                detections.append(self._build_detection(frame, x1, y1, x2, y2, float(cls_confs[k]), int(cid)))

        return detections

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

        u, v, pts = extract_bottom_point_by_contour(
            frame,
            x1 - BOX_PADDING, y1 - BOX_PADDING,
            x2 + BOX_PADDING, y2 + BOX_PADDING
        )

        return {
            "bbox": (int(x1), int(y1), int(bw), int(bh)),
            "center_u": float(u),
            "bottom_v": float(v),
            "had_contour": pts is not None,
            "world_xy": None,
            "class_name": self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id),
            "class_id": int(cls_id),
            "conf": float(conf),
            "contour_pts": pts,
        }

# -----------------------------
# Track / Tracker
# -----------------------------
class Track:
    def __init__(self, track_id, det, timestamp):
        self.id = track_id
        self.smoothed_depth = np.nan
        self.missed_frames = 0
        self.last_update = timestamp

        self.bbox = list(det["bbox"])
        self.center_u = float(det["center_u"])
        self.bottom_v = float(det["bottom_v"])
        self.world_xy = det["world_xy"] if det["world_xy"] is not None else (np.nan, np.nan)
        self.class_name = det["class_name"]
        self.source = "contour" if det["had_contour"] else "bbox"
        self.contour_pts = det.get("contour_pts", None)

    def get_predicted_bbox(self):
        return tuple(map(int, self.bbox))

    def update(self, det, timestamp):
        alpha = 0.25
        for i in range(4):
            self.bbox[i] = alpha * det["bbox"][i] + (1.0 - alpha) * self.bbox[i]
        self.center_u = alpha * det["center_u"] + (1.0 - alpha) * self.center_u
        self.bottom_v = alpha * det["bottom_v"] + (1.0 - alpha) * self.bottom_v

        if det["world_xy"] is not None:
            self.world_xy = det["world_xy"]

        self.class_name = det["class_name"] or self.class_name
        self.source = "contour" if det["had_contour"] else "bbox"
        self.contour_pts = det.get("contour_pts", None)
        self.missed_frames = 0
        self.last_update = timestamp

    def recompute_depth(self, ipm):
        try:
            xy = ipm.uv2xy(np.array([[self.center_u], [self.bottom_v]], dtype=np.float64))
            new_depth = float(xy[1, 0]) / 1000.0
            if np.isfinite(new_depth):
                alpha_d = 0.05
                self.smoothed_depth = (
                    new_depth if np.isnan(self.smoothed_depth)
                    else alpha_d * new_depth + (1.0 - alpha_d) * self.smoothed_depth
                )
        except Exception:
            pass

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
                [iou_xyxy(self._bbox_xyxy_from_xywh(self.tracks[tid].get_predicted_bbox()),
                          self._bbox_xyxy_from_xywh(d["bbox"])) for d in detections]
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
            self.tracks[self.next_id] = Track(self.next_id, detections[di], timestamp)
            self.next_id += 1

        for tid in [tid for tid, tr in self.tracks.items() if tr.missed_frames > self.max_missed]:
            del self.tracks[tid]

        return self.tracks

# -----------------------------
# Main loop
# -----------------------------
def main(yield_every_frame=False):
    cv2.setNumThreads(0)
    cv2.setUseOptimized(True)

    with open(CONFIG["names"], "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f]

    detector = YOLOv8ONNX(
        CONFIG["onnx"],
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
        "camera_height": 750.0,
        "pitch": 90.0,
        "yaw": 0.0,
    })
    ipm = IPM(cam)

    # --- Inside main() ---
    vs = WebcamVideoStream(src=CAMERA_INDEX, width=640, height=480, fps=60).start()

    tracker = Tracker(max_missed=MAX_MISSED_FRAMES, match_iou=MATCH_IOU_THRESH)
    frame_count = 0
    fps_ema = 0.0

    try:
        while True:
            t_start = time.time()

            frame = vs.read() # This is now non-blocking and instant
            if frame is None:
                break

            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            frame = cv2.remap(frame, UNDISTORT_MAP1, UNDISTORT_MAP2, interpolation=cv2.INTER_LINEAR)

            frame_count += 1

            detections = []
            if frame_count % DETECTION_INTERVAL == 0:
                detections = detector.detect(frame, target_classes=TARGET_CLASSES)

                for d in detections:
                    uv = np.array([[d["center_u"]], [d["bottom_v"]]], dtype=np.float64)
                    xy = ipm.uv2xy(uv)
                    d["world_xy"] = tuple(float(v) for v in xy.reshape(-1))

            tracks = tracker.step(detections, time.time())

            vis = frame.copy()

            for tid, tr in tracks.items():
                tr.recompute_depth(ipm)

                x, y, w, h = tr.get_predicted_bbox()
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis, (int(tr.center_u), int(tr.bottom_v)), 5, (0, 0, 255), -1)

                if tr.contour_pts is not None:
                    cv2.drawContours(vis, [tr.contour_pts.astype(int)], -1, (0, 180, 180), 2)

                depth_text = f"{tr.smoothed_depth:.1f}m" if np.isfinite(tr.smoothed_depth) else "na"
                label = f"ID:{tr.id} {tr.class_name} {depth_text}"
                cv2.putText(vis, label, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            dt = time.time() - t_start
            if dt > 0:
                cur_fps = 1.0 / dt
                fps_ema = cur_fps if fps_ema == 0 else (0.9 * fps_ema + 0.1 * cur_fps)

            #cv2.putText(vis, f"FPS: {fps_ema:.1f}", (10, 30),
                       # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            #cv2.imshow("Pi 4 YOLOv8 INT8 ONNX", vis)

            if yield_every_frame:
                yield vis, tracks  # Yield the visualization frame AND the tracks

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # --- Change this in your finally block ---
    finally:
        vs.stop()  # Correct way to shut down the threaded stream
        cv2.destroyAllWindows()

if __name__ == "__main__":
    for _ in main(False):
        pass