TASK - 3:

1.  Flask API script for real-time object detection

Overview:
This project implements a Flask-based REST API for real-time object detection in videos using the YOLOv5 model. The API allows users to upload a video file, process each frame using YOLOv5, and return the detected objects with bounding boxes.

Features:
Uses YOLOv5 for object detection.
Processes uploaded video files frame by frame.
Draws bounding boxes around detected objects.
Returns processed frames as a response.

Prerequisites:
Flask
Flask-RESTful
OpenCV (cv2)
Torch (PyTorch)
torchvision
YOLOv5 repository

Usage:
Your Flask API starts using the following command: python app.py
This executes the app.py script, which contains:
if __name__ == '__main__':
    app.run(debug=True)

How It Works:
if __name__ == '__main__': ensures the script runs only when executed directly.
app.run(debug=True) starts the Flask development server at http://127.0.0.1:5000/.
The debug=True flag enables auto-reload for code changes and provides detailed error messages.


Notes:
This API is for development and testing. For production, use a WSGI server.


2. Why Optimize for Edge Devices?

Edge devices (like IoT sensors, drones, or mobile phones) often have limited computing power, less memory, lower energy capacity. If we try to run a large, unoptimized model, the device may struggle or even fail to run it. Optimization makes the model smaller, faster, and more efficient, so it can run smoothly even on low-powered devices.

Strategies for Optimizing Models
Model Pruning - Think of pruning like trimming a tree i.e., remove weak or unused branches (neurons/weights) that don’t contribute much to accuracy.
Result: Smaller, faster model with minimal accuracy loss.

Knowledge Distillation (Teaching a smaller model) - Train a smaller "student" model to mimic the larger "teacher" model. The smaller model learns the most important patterns, so it can make predictions almost as well as the big model but much faster.

Model Quantization (Reducing precision) - Models usually use 32-bit floating-point numbers (high precision but slow). Quantization reduces precision (e.g., to 8 bits), which speeds up computations and shrinks the model size.

Efficient Architectures (Choosing smaller model designs) - Use lightweight architectures like MobileNet, EfficientNet, or Tiny-YOLO, which are designed to be fast and small, perfect for edge devices.

TensorFlow Lite & ONNX (Optimized formats for deployment) - Convert your model to formats like TensorFlow Lite (TFLite) or ONNX, which are built for running models on edge devices.

Real-World Use Cases:
Security cameras detecting people or objects in real time.
Smart home devices recognizing voice commands or gestures.
Drones detecting obstacles while flying autonomously.
Wearables monitoring health metrics without draining the battery.