import numpy as np
import time

ONNX_PATH = '/home/adji/ttc_project/models/yolov8n_int8_static_safe.onnx'
NCNN_PARAM = '/home/adji/ttc_project/yolov8n_ncnn_model/model.ncnn.param'
NCNN_BIN   = '/home/adji/ttc_project/yolov8n_ncnn_model/model.ncnn.bin'
WARMUP = 5
RUNS   = 30

def bench_ort():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    input_name = inp.name
    _, _, h, w = inp.shape
    h, w = int(h), int(w)
    print(f"[ORT]  input: {w}x{h}")

    dummy = np.random.rand(1, 3, h, w).astype(np.float32)

    for _ in range(WARMUP):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"[ORT]  avg: {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
    return avg_ms

def bench_ncnn():
    import ncnn
    net = ncnn.Net()
    net.opt.num_threads = 4
    net.opt.use_vulkan_compute = False
    net.load_param(NCNN_PARAM)
    net.load_model(NCNN_BIN)

    size = 320
    print(f"[NCNN] input: {size}x{size}")

    dummy = (np.random.rand(size, size, 3) * 255).astype(np.uint8)

    def run_once():
        mat = ncnn.Mat.from_pixels(dummy, ncnn.Mat.PixelType.PIXEL_RGB, size, size)
        mat.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
        ex = net.create_extractor()
        ex.input("in0", mat)
        ex.extract("out0")

    for _ in range(WARMUP):
        run_once()

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"[NCNN] avg: {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
    return avg_ms

print("=== Benchmarking ORT ===")
ort_ms = bench_ort()

print("\n=== Benchmarking NCNN ===")
ncnn_ms = bench_ncnn()

print(f"\n=== Result ===")
print(f"NCNN is {ort_ms/ncnn_ms:.1f}x faster than ORT")
