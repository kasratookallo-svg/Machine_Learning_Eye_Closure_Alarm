# Machine Learning Eye Closure Alarm

A real-time computer vision system that detects whether a person's eyes are **open** or **closed** using a webcam, MediaPipe Face Mesh, and a trained Random Forest classifier. When closed eyes are detected continuously for about **one second**, the application triggers an audio alarm.

## 🚨 Features

- 🎥 Real-time webcam inference
- 👁️ Facial landmark detection with **MediaPipe Face Mesh**
- 📐 Geometric eye-feature extraction
- 🤖 Random Forest classification (`OPEN` / `CLOSED`)
- 🧠 Majority-vote temporal smoothing over a 5-frame window
- ⏰ Alarm trigger after ~1 second of continuous closed-eye detection
- 📊 On-screen status overlay (prediction, probabilities, EAR, alarm state)

## 🔄 How It Works

1. **Capture** frames from the webcam.
2. **Detect** facial landmarks using MediaPipe Face Mesh.
3. **Extract** 9 eye-related geometric features.
4. **Classify** the eye state with a pretrained Random Forest model.
5. **Smooth** predictions with a 5-frame majority vote.
6. **Start** a closed-eye timer when the state is `CLOSED`.
7. **Trigger** an audio alarm if closure lasts about 1 second.
