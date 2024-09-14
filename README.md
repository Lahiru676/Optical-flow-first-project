# Optical-flow-first-project
learn about optical flow beheviors


This script demonstrates real-time object movement analysis using optical flow with a laptop webcam. Here's a brief explanation of how it works:

It captures video from the webcam using OpenCV.
For each frame, it calculates the optical flow using the Farneback method, which computes the movement between the current and previous frame.
The optical flow is visualized using color coding: the hue represents the direction of movement, while the intensity represents the magnitude of movement.
The original frame and the optical flow visualization are displayed side by side.