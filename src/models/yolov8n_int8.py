import os
import glob
import cv2
import numpy as np

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

FP32_MODEL = r"C:\Users\adjip\Documents\python\ASO-IPM\src\models\yolov8n.onnx"
INT8_MODEL = r"C:\Users\adjip\Documents\python\ASO-IPM\src\models\yolov8n_int8.onnx"
CALIB_DIR = r"C:\Users\adjip\Documents\python\ASO-IPM\src\data\calibration_int8"  # Folder with your 640x480 calibration images
INPUT_SIZE = 320


def letterbox(im, new_shape=320, color=(114, 114, 114), scaleup=False):
    shape = im.shape[:2]  # h, w
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
    return im


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = letterbox(img, INPUT_SIZE, scaleup=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name):
        self.image_paths = image_paths
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        while self.index < len(self.image_paths):
            path = self.image_paths[self.index]
            self.index += 1
            blob = preprocess_image(path)
            if blob is None:
                continue
            return {self.input_name: blob}
        return None


def main():
    if not os.path.exists(FP32_MODEL):
        raise FileNotFoundError(FP32_MODEL)

    image_paths = sorted(
        glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
        + glob.glob(os.path.join(CALIB_DIR, "*.jpeg"))
        + glob.glob(os.path.join(CALIB_DIR, "*.png"))
    )
    if not image_paths:
        raise RuntimeError(f"No calibration images found in: {CALIB_DIR}")

    # Read input name from the model
    import onnxruntime as ort
    sess = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    reader = YOLOCalibrationDataReader(image_paths, input_name)

    quantize_static(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    print(f"Saved INT8 model to: {INT8_MODEL}")


if __name__ == "__main__":
    main()