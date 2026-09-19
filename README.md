# Machine Learning Eye Closure Alarm 👁️🚨

A real-time Computer Vision system designed to monitor eye state (Open/Closed) and trigger an alarm to prevent fatigue-related incidents (e.g., while driving or studying).

## 🚀 Features

- **Real-time Inference:** Efficient processing using your webcam.
- **MediaPipe Integration:** Utilizes `MediaPipe Face Mesh` for highly accurate facial landmark detection.
- **Geometric Feature Engineering:** Extracts 9 precise eye-related geometric features rather than relying on raw pixels.
- **Machine Learning Powered:** Classification performed by a trained `Random Forest` model.
- **Temporal Smoothing:** Implements a 5-frame "majority vote" window to prevent flickering and false alarms.
- **Visual & Audio Feedback:** On-screen status overlay (EAR, prediction, alarm status) + audible alert.

## 🛠 Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install opencv-python mediapipe scikit-learn numpy

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

## 🤖 Model Training Strategy
We utilize a dual-model approach to ensure reliability:
1. **Random Forest Classifier (Primary):** Selected for its high performance on structured geometric data and interpretability. We perform grid search to optimize hyperparameters like `n_estimators` and `max_depth`.
2. **Support Vector Machine (Secondary/Comparison):** Trained in parallel to compare baseline performance. We use the SVM for cross-validation to ensure the Random Forest is not overfitting to specific lighting conditions.

*This approach allows us to choose the most robust classifier for real-time inference.*
