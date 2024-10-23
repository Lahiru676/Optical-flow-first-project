import cv2
import numpy as np
import time

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Get the frame rate (fps) of the camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # If the camera does not provide fps, use time-based calculation
        fps = 30  # Assume a default fps of 30
    time_interval = 1 / fps

    # Read the first frame
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    # Set the optical flow threshold to identify vehicles
    motion_threshold = 2.0  # Adjust to control sensitivity to vehicle motion

    while True:
        # Record the time at the start of the frame
        start_time = time.time()

        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of the 2D flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold the magnitude to isolate significant motion (vehicles)
        motion_mask = magnitude > motion_threshold

        # Find contours of the moving regions (vehicles)
        mask = np.uint8(motion_mask * 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicle_detected = False
        for contour in contours:
            # Filter small contours that are not vehicles
            if cv2.contourArea(contour) < 500:
                continue

            # Vehicle detected
            vehicle_detected = True

            # Create a bounding box around the vehicle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the average flow magnitude within the bounding box (vehicle speed)
            avg_magnitude = np.mean(magnitude[y:y+h, x:x+w])

            # Convert magnitude to speed (speed = distance/time)
            speed = avg_magnitude / time_interval

            # Display the speed on the frame
            cv2.putText(frame, f"Speed: {speed:.2f} units/sec", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not vehicle_detected:
            # If no vehicle is detected, display 'No Vehicle Detected'
            cv2.putText(frame, "No Vehicle Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update previous frame for optical flow calculation
        prev_gray = gray

        # Display the original frame with vehicle bounding boxes and speeds
        cv2.imshow('Vehicle Detection and Speed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Adjust the time interval based on actual frame processing time
        end_time = time.time()
        processing_time = end_time - start_time
        time_interval = max(1 / fps, processing_time)  # To avoid division by zero

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
