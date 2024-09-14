import cv2
import numpy as np

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Read the first frame
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of 2D vector
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Set hue according to the angle of optical flow
        hsv[..., 0] = angle * 180 / np.pi / 2

        # Set value according to the normalized magnitude of optical flow
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Display the original frame and the optical flow
        cv2.imshow('Original', frame)
        cv2.imshow('Optical Flow', rgb)

        # Update previous frame
        prev_gray = gray

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
