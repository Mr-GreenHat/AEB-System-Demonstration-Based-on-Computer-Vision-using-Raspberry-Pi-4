import ncnn
import numpy as np
import time

PARAM = '/home/adji/ttc_project/yolov8n_ncnn_model/model.ncnn.param'
BIN   = '/home/adji/ttc_project/yolov8n_ncnn_model/model.ncnn.bin'
INPUT_SIZE = 320
CONF_THRESH = 0.25

net = ncnn.Net()
net.opt.num_threads = 4
net.opt.use_vulkan_compute = False
net.load_param(PARAM)
net.load_model(BIN)
print('model loaded ok')

# dummy frame (replace with real camera frame later)
dummy_frame = (np.random.rand(INPUT_SIZE, INPUT_SIZE, 3) * 255).astype(np.uint8)

def infer(frame_rgb):
    mat_in = ncnn.Mat.from_pixels(frame_rgb, ncnn.Mat.PixelType.PIXEL_RGB, INPUT_SIZE, INPUT_SIZE)
    mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])

    ex = net.create_extractor()
    ex.input("in0", mat_in)
    _, out = ex.extract("out0")

    out = np.array(out)  # [84, 2100]
    boxes_xywh = out[:4].T   # [2100, 4]
    scores = out[4:].T        # [2100, 80]

    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    detections = []
    for i, conf in enumerate(confs):
        if conf > CONF_THRESH:
            cx, cy, w, h = boxes_xywh[i]
            detections.append((cx, cy, w, h, float(conf), int(class_ids[i])))
    return detections

# warmup
infer(dummy_frame)

# benchmark 30 frames
times = []
for _ in range(30):
    t0 = time.perf_counter()
    dets = infer(dummy_frame)
    times.append(time.perf_counter() - t0)

avg_ms = np.mean(times) * 1000
print(f"avg inference: {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
print(f"detections on dummy frame: {len(dets)}")
