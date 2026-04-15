import cv2
import os
import numpy as np

# ===== SETTINGS =====
camera_index = 1
width = 640
height = 480
# Path to your calibration file
intrinsics_path = r"C:\Users\adjip\Documents\python\ASO-IPM\calibration\camera_intrinsics640x480.npz"

# ===== LOAD CALIBRATION & PRECOMPUTE MAPS =====
if os.path.exists(intrinsics_path):
    data = np.load(intrinsics_path)
    mtx = data["K"]
    dist = data["dist"]
    print("Calibration loaded successfully.")
else:
    # Fallback if file is missing (Use your real values here)
    print("Warning: Calibration file not found. Using identity matrix.")
    mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

# Precompute the maps for speed (removes distortion "stretch" math from the loop)
map1, map2 = cv2.initUndistortRectifyMap(
    mtx, dist, None, mtx, (width, height), cv2.CV_16SC2
)

# ===== INIT CAMERA =====
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

img_count = 0
print("System Ready. Press 'c' to capture undistorted, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. APPLY UNDISTORTION IN REAL-TIME
    # This transforms the "fish-eye" look into a linear perspective
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 2. DISPLAY THE CORRECTED VIEW
    cv2.imshow("Undistorted Preview", undistorted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        filename = f"undistorted_{img_count}.png"
        # 3. SAVE THE CORRECTED FRAME
        cv2.imwrite(filename, undistorted_frame)
        print(f"Saved Undistorted: {os.path.abspath(filename)}")
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()