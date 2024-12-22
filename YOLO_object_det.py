
#Download the required dataset and weights

# wget https://pjreddie.com/media/files/yolov3.weights
# wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O yolov3.cfg
# wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O coco.names


import numpy as np
import cv2
import time

# Load the YOLO model and configuration weights
model_network = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as data_frame:
    classes = [line.strip() for line in data_frame.readlines()]
layer_names = model_network.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture(0)

intruder_classes = ['person', 'car', 'bus', 'truck']

# Initialize variables
snapshot_count = 0
count_limit = 3
snapshot_shuttered = []
exit_flag = False

while True:
    ret, frame = cap.read()

    # If the camera feed is unavailable, break the loop
    if not ret:
        print("Failed to initiate camera feed")
        break

    # Perform YOLO object detection - pre-processing
    height, width, _ = frame.shape

    # Scale and normalize the input frame to the required format
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model_network.setInput(blob)
    final_layer_output = model_network.forward(layer_names)

    # Post-process YOLO outputs
    class_ids = []
    confidences = []
    boxes = []

    for out in final_layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove duplicate bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        if isinstance(i, list):
            i = i[0]
        box = boxes[i]
        x, y, w, h = box
        class_id = class_ids[i]
        label = f"{classes[class_id]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect only intruder classes (e.g., person, car)
        if classes[class_id] in intruder_classes:
            # Take a snapshot
            if snapshot_count < count_limit:
                snapshot_filename = f"intruder_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_filename, frame)
                snapshot_shuttered.append(snapshot_filename)
                snapshot_count += 1
                print(f"Snapshot taken: {snapshot_filename}")

            if snapshot_count >= count_limit:
                print("Maximum snapshots taken.")
                exit_flag = True
                break

    # Display the result
    cv2.imshow("Intruder Detection", frame)

    # Break the loop if 'q' key is pressed or exit_flag is True
    if cv2.waitKey(1) & 0xFF == ord('q') or exit_flag:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Print all snapshots taken
print("Snapshots saved:", snapshot_shuttered)
