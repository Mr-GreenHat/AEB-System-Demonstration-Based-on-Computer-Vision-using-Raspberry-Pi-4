# input image 640x320
import numpy as np
import cv2

# paths
weights = "src\\models\\yolov3-tiny.weights"
config = "src\\config\\yolov3-tiny.cfg"
names = "src\\coco.names"

# load class names
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# load network
net = cv2.dnn.readNet(weights, config)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# read image
img = cv2.imread("src\\ipm\\test_img.png")
img = cv2.resize(img, (1080,720))
height, width = img.shape[:2]

# create blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
net.setInput(blob)

# run detection
outs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

# parse detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# draw results
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]

        # bounding box
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # bottom line of bounding box (ground contact)
        cv2.line(img, (x, y+h), (x+w, y+h), (0,0,255), 2)

        label = classes[class_ids[i]]
        cv2.putText(img, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# show result
cv2.imshow("detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()