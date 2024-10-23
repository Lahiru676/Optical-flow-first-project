import cv2
import numpy as np

# Load YOLO model from OpenCV's DNN module
def load_yolo_model():
    # Load the pre-trained YOLOv3 model from weights and cfg
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Function to detect vehicles using YOLO
def detect_vehicles(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Lists to store detected bounding boxes
    boxes = []
    confidences = []
    class_ids = []

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for confidence
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Only consider class IDs for 'car', 'truck', 'bus'
                if class_id in [2, 5, 7]:  # COCO dataset labels: 2=car, 5=bus, 7=truck
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Non-max suppression
    vehicle_boxes = [boxes[i] for i in indices.flatten()]
    return vehicle_boxes

def main():
    # Load YOLO model
    net, output_layers = load_yolo_model()

    # Open the webcam or video file
    cap = cv2.VideoCapture(0)

    # Get the frame rate (fps) of the camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # If the camera does not provide fps, use time-based calculation
        fps = 30  # Assume a default fps of 30
    time_interval = 1 / fps

    # Read the first frame
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read a new frame and resize to reduce processing time
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles using YOLO
        vehicle_boxes = detect_vehicles(frame, net, output_layers)

        # Convert current frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in vehicle_boxes:
            # Crop the region containing the vehicle
            vehicle_roi_prev = prev_gray[y:y+h, x:x+w]
            vehicle_roi_curr = gray[y:y+h, x:x+w]

            # Calculate optical flow within the bounding box (vehicle region)
            flow = cv2.calcOpticalFlowFarneback(vehicle_roi_prev, vehicle_roi_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate the average magnitude (which relates to speed)
            avg_magnitude = np.mean(magnitude)

            # Convert magnitude to speed (arbitrary units)
            speed = avg_magnitude / time_interval

            # Draw bounding box and speed on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Speed: {speed:.2f} units/sec", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update previous frame for the next optical flow calculation
        prev_gray = gray

        # Display the frame
        cv2.imshow('Vehicle Detection and Speed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
