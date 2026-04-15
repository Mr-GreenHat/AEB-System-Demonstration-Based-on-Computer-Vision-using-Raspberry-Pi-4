import cv2
import os

# ===== SETTINGS =====
camera_index = 1
width = 640
height = 480

# ===== INIT CAMERA =====
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

img_count = 0

print("Press 'c' to capture, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f"img_{img_count}.png"  # saved in current folder
        cv2.imwrite(filename, frame)
        print(f"Saved: {os.path.abspath(filename)}")
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()