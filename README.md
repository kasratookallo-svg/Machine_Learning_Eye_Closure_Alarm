# Real-Time Eye Open / Closed Detection

A real-time eye state detection system using **OpenCV**, **MediaPipe Face Mesh**, and **Random Forest**.

The system detects whether the eyes are:

- OPEN
- CLOSED

It processes webcam frames in real time, extracts geometric eye features from MediaPipe facial landmarks, classifies the eye state using a trained Random Forest model, applies temporal smoothing using majority voting, and triggers an audio alarm when the eyes remain closed for approximately one second.

---

## 1. Project Overview

The objective of this project is to develop a real-time computer vision system capable of detecting eye closure from a standard webcam.

The project combines:

- Computer Vision
- Facial Landmark Detection
- Feature Engineering
- Machine Learning
- Real-Time Classification
- Temporal Signal Processing
- Audio Alert

The system is designed as a modular pipeline:

Webcam → Face Mesh → Eye Landmarks → Feature Extraction → Random Forest → Temporal Filtering → Alarm

---

## 2. System Architecture

```text
                     ┌─────────────────┐
                     │     Webcam      │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │     OpenCV      │
                     │ Frame Capture   │
                     └────────┬────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │  MediaPipe Face Mesh │
                  │  Facial Landmarks    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Eye Landmark       │
                  │   Extraction         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Extraction   │
                  │                      │
                  │ 9 Features           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Random Forest      │
                  │    Classifier        │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ OPEN / CLOSED   │
                    └────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Temporal Filtering   │
                  │ Majority Voting      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Closed-Eye Timer     │
                  │ ~ 1 Second           │
                  └──────────┬───────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │ Audio Alarm  │
                     └──────────────┘
