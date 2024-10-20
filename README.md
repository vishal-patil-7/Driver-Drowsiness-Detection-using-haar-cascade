# Drowsiness Detection System

This project implements a drowsiness detection system using OpenCV, dlib, and imutils. It monitors the driver's eyes in real-time and triggers an alert if drowsiness is detected.

## How It Works
- The system detects the face and eye landmarks using Haar Cascades and dlib’s facial landmark detector.
- It calculates the Eye Aspect Ratio (EAR) to determine if the driver's eyes have been closed for an extended period, indicating drowsiness.
- When the EAR drops below a threshold for a certain number of frames, the system triggers:
  - An SMS alert to a designated phone number.
  - An email alert with a screenshot of the driver’s face.

## Features
- **Real-Time Monitoring**: Uses a live video feed from a camera (Raspberry Pi camera or webcam).
- **Alerts**: Sends SMS and email alerts when drowsiness is detected.
- **Threshold-Based Detection**: Detects eye closure over consecutive frames to reduce false positives.

## Prerequisites
- OpenCV
- dlib
- imutils
- smtplib (for email alerts)
- urllib (for sending SMS via MSG91 API)

## Running the Code
1. Clone the repository and navigate to the project directory.
2. Run the script with the following command:
   ```bash
   python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
