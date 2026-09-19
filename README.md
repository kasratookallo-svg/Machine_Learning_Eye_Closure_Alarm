# Machine Learning Eye Closure Alarm 👁️🚨

A real-time Computer Vision system designed to monitor eye state (Open/Closed) and trigger an alarm to prevent fatigue-related incidents (e.g., while driving or studying).

## 🚀 Features

- **Real-time Inference:** Efficient processing using your webcam.
- **MediaPipe Integration:** Utilizes `MediaPipe Face Mesh` for highly accurate facial landmark detection.
- **Geometric Feature Engineering:** Extracts 9 precise eye-related geometric features rather than relying on raw pixels.
- **Machine Learning Powered:** Classification performed by a trained `Random Forest` model.
- **Temporal Smoothing:** Implements a 5-frame "majority vote" window to prevent flickering and false alarms.
- **Visual & Audio Feedback:** On-screen status overlay (EAR, prediction, alarm status) + audible alert.

## 🔬 Technical Deep Dive

## 📐 9-Feature Geometry
Instead of relying on raw pixel data, we extract 9 geometric features that capture the spatial relationship of facial landmarks. These features provide robustness against variations in lighting and head orientation:
- **Eye Aspect Ratio (EAR):** The primary indicator of eye openness.
- **Euclidean Distances:** Ratios between vertical and horizontal eyelid landmarks.
- **Normalized Landmarks:** Relative coordinates to ensure scale invariance.
- (Additional custom features calculated via MediaPipe Face Mesh to differentiate between squinting and closing).

## 🧹 Outlier & Suspicious Samples
We have implemented a data cleaning pipeline to ensure model robustness:
- **Outlier Detection:** Filtering out frames with low confidence scores from MediaPipe or extreme geometric values (e.g., face occlusions, rapid motion blur).
- **Suspicious Samples:** Identifying and removing frames where landmark detection is inconsistent or logically impossible (e.g., implausible aspect ratios), ensuring the Random Forest only trains on clean, reliable data.

## 🧠 Model Training Strategy
To understand the impact of data quality on predictive performance, we trained two separate **Random Forest** models using different datasets:

1. **Model A (Clean Dataset - 995 Samples):** Trained on a curated dataset where outliers and suspicious samples were filtered out. This model focuses on high precision and reliability.
2. **Model B (Full Dataset - 1000 Samples):** Trained on the raw dataset including all samples. This serves as a baseline to quantify the performance gain achieved through our data-cleaning pipeline.

*This comparison allows us to validate the effectiveness of our pre-processing steps and ensures that our final real-time model is not biased by noisy or unreliable data points.*


*This strategy enables us to select the most robust model for real-time inference and provides a reliable baseline for future improvements.*



## 🛠 Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install opencv-python mediapipe scikit-learn numpy

