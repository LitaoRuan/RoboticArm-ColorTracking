# RoboticArm-ColorTracking
This project implements a real-time computer vision system using a Raspberry Pi that tracks a colored object and aligns servos using a PID controller. It uses OpenCV for image processing and the HiWonder BusServo SDK for hardware control.
# Features
**Color Detection**: Detects objects of a specific color (default: green) in a video stream using LAB color space.<br>
**PID-Controlled Servo Alignment**: Continuously adjusts servo motors to align the robotic arm to the detected object's center.<br>
**Coordinate Mapping**: Converts pixel coordinates to real-world coordinates.<br>
**Modular Design**: Cleanly separated into PID control, hardware abstraction, and image tracking logic.<br>
# How It Works
**1.Image Capture:**<br>
·The camera captures frames using OpenCV.<br>
**2.Image Processing:**<br>
·Convert frames to LAB color space.<br>
·Apply Gaussian blur and morphological transformations.<br>
·Detect contours of the specified color.<br>
**3.Object Tracking:**<br>
·Identify the largest colored object.<br>
·Draw a bounding box and compute its center.<br>
·Convert center coordinates to real-world space.<br>
**4.Servo Adjustment:**<br>
·Apply PID control based on the distance of the object from the center of the image.<br>
·Move servos incrementally to keep the object centered.<br>
# Running the Code
```bash
python3 colorTracking.py
```
# Configuration
```python3
detect_color = 'green'  # Options: 'red', 'green', 'blue', etc.

# PID tuning
x_pid = PID.PID(P=0.1, I=0.00, D=0.008)
```
# Key Modules
**PID.py**<br>
Implements a basic PID controller to calculate precise control signals based on the current and desired position of the object.<br>

**colorTracking.py**<br>
Main logic for:<br>
·Reading video input<br>
·Applying image transformations<br>
·Detecting and tracking colored objects<br>
·Controlling servos via the Board module<br>
**Board.py**<br>
Abstracts low-level communication with the HiWonder bus servo motors using I2C. Also provides LED and motor functions.<br>
