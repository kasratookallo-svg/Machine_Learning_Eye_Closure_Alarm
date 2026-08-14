import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import winsound


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "eye_random_forest_v2.pkl"

FEATURES = [
    "left_ear",
    "right_ear",
    "ear",
    "left_eye_width",
    "left_eye_height_1",
    "left_eye_height_2",
    "right_eye_width",
    "right_eye_height_1",
    "right_eye_height_2"
]

TEMPORAL_WINDOW = 5

CLOSED_DURATION = 1.0

ALARM_FREQUENCY = 1500
ALARM_DURATION = 700


# ============================================================
# DISPLAY COLOR
# OpenCV uses BGR
# GREEN = (0, 255, 0)
# ============================================================

TEXT_COLOR = (0, 255, 0)


# ============================================================
# MEDIAPIPE EYE LANDMARKS
# ============================================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# FEATURE EXTRACTION
# NORMALIZED MEDIAPIPE COORDINATES
# ============================================================

def extract_features(landmarks):

    points = []

    for landmark in landmarks:

        points.append(
            (
                landmark.x,
                landmark.y
            )
        )

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_362 = points[362]
    left_385 = points[385]
    left_387 = points[387]
    left_263 = points[263]
    left_373 = points[373]
    left_380 = points[380]

    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    right_33 = points[33]
    right_160 = points[160]
    right_158 = points[158]
    right_133 = points[133]
    right_153 = points[153]
    right_144 = points[144]

    # --------------------------------------------------------
    # WIDTH
    # --------------------------------------------------------

    left_eye_width = distance(
        left_362,
        left_263
    )

    right_eye_width = distance(
        right_33,
        right_133
    )

    # --------------------------------------------------------
    # HEIGHT
    # --------------------------------------------------------

    left_eye_height_1 = distance(
        left_385,
        left_380
    )

    left_eye_height_2 = distance(
        left_387,
        left_373
    )

    right_eye_height_1 = distance(
        right_160,
        right_144
    )

    right_eye_height_2 = distance(
        right_158,
        right_153
    )

    # --------------------------------------------------------
    # EAR
    # --------------------------------------------------------

    left_ear = (
        left_eye_height_1 +
        left_eye_height_2
    ) / (
        2.0 * left_eye_width
    )

    right_ear = (
        right_eye_height_1 +
        right_eye_height_2
    ) / (
        2.0 * right_eye_width
    )

    ear = (
        left_ear +
        right_ear
    ) / 2.0

    return {

        "left_ear":
            left_ear,

        "right_ear":
            right_ear,

        "ear":
            ear,

        "left_eye_width":
            left_eye_width,

        "left_eye_height_1":
            left_eye_height_1,

        "left_eye_height_2":
            left_eye_height_2,

        "right_eye_width":
            right_eye_width,

        "right_eye_height_1":
            right_eye_height_1,

        "right_eye_height_2":
            right_eye_height_2
    }


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("EYE CLOSURE ALARM - RANDOM FOREST V2")
print("=" * 70)

print()
print("Loading model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")

print()
print("Classes:")
print(model.classes_)

print()
print("Trees:")
print(model.n_estimators)


# ============================================================
# SINGLE THREAD PREDICTION
# ============================================================

model.set_params(
    n_jobs=1
)

print()
print("Real-time n_jobs:")
print(model.n_jobs)

print()
print("Temporal window:")
print(TEMPORAL_WINDOW)

print()
print("Closed duration:")
print(f"{CLOSED_DURATION:.1f} seconds")

print()
print("Alarm:")
print("Enabled")


# ============================================================
# OPEN WEBCAM
# ============================================================

print()
print("Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Cannot open webcam."
    )


# ============================================================
# MEDIAPIPE
# ============================================================

face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# TEMPORAL BUFFER
# ============================================================

prediction_buffer = []


# ============================================================
# CLOSED TIMER
# ============================================================

closed_start_time = None

alarm_triggered = False


# ============================================================
# COUNTERS
# ============================================================

total_frames = 0

face_frames = 0


# ============================================================
# ALARM FUNCTION
# ============================================================

def play_alarm():

    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ALARM: EYES CLOSED FOR 1 SECOND")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()

    winsound.Beep(
        ALARM_FREQUENCY,
        ALARM_DURATION
    )


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 70)
print("REAL-TIME EYE ALARM STARTED")
print("=" * 70)

print()
print("Keep the webcam running.")
print("Press ESC or Q to quit.")
print()


while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Cannot read webcam frame."
        )

        break

    total_frames += 1

    # --------------------------------------------------------
    # MIRROR
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    results = face_mesh.process(
        rgb
    )

    raw_prediction = None

    temporal_prediction = None

    confidence = 0.0

    closed_probability = 0.0

    open_probability = 0.0

    closed_time = 0.0

    feature_dict = None


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if results.multi_face_landmarks:

        face_frames += 1

        face_landmarks = (
            results.multi_face_landmarks[0]
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        feature_dict = extract_features(
            face_landmarks.landmark
        )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        X_live = pd.DataFrame(

            [[
                feature_dict[feature]
                for feature in FEATURES
            ]],

            columns=FEATURES
        )

        # ----------------------------------------------------
        # RAW PREDICTION
        # ----------------------------------------------------

        raw_prediction = model.predict(
            X_live
        )[0]

        probabilities = model.predict_proba(
            X_live
        )[0]

        class_probabilities = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        open_probability = (
            class_probabilities.get(
                "OPEN",
                0.0
            )
        )

        closed_probability = (
            class_probabilities.get(
                "CLOSED",
                0.0
            )
        )

        confidence = max(
            probabilities
        )

        # ----------------------------------------------------
        # TEMPORAL BUFFER
        # ----------------------------------------------------

        prediction_buffer.append(
            raw_prediction
        )

        if len(prediction_buffer) > TEMPORAL_WINDOW:

            prediction_buffer.pop(0)

        # ----------------------------------------------------
        # MAJORITY VOTING
        # ----------------------------------------------------

        if len(prediction_buffer) > 0:

            closed_votes = prediction_buffer.count(
                "CLOSED"
            )

            open_votes = prediction_buffer.count(
                "OPEN"
            )

            if closed_votes > open_votes:

                temporal_prediction = "CLOSED"

            else:

                temporal_prediction = "OPEN"

        # ====================================================
        # CLOSED TIMER
        # ====================================================

        current_time = time.time()

        if temporal_prediction == "CLOSED":

            # ------------------------------------------------
            # START TIMER
            # ------------------------------------------------

            if closed_start_time is None:

                closed_start_time = current_time

            # ------------------------------------------------
            # CALCULATE CLOSED DURATION
            # ------------------------------------------------

            closed_time = (
                current_time -
                closed_start_time
            )

            # ------------------------------------------------
            # ALARM
            # ------------------------------------------------

            if (
                closed_time >= CLOSED_DURATION
                and not alarm_triggered
            ):

                play_alarm()

                alarm_triggered = True

        else:

            # ------------------------------------------------
            # EYES OPEN
            # RESET TIMER
            # ------------------------------------------------

            closed_start_time = None

            closed_time = 0.0

            # ------------------------------------------------
            # RE-ARM ALARM
            # ------------------------------------------------

            alarm_triggered = False

    else:

        # ====================================================
        # NO FACE
        # ====================================================

        temporal_prediction = None

        prediction_buffer.clear()

        closed_start_time = None

        closed_time = 0.0

        alarm_triggered = False


    # ========================================================
    # DISPLAY
    # ========================================================

    if temporal_prediction is not None:

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"STATE: {temporal_prediction}",

            (20, 45),

            cv2.FONT_HERSHEY_DUPLEX,

            0.9,

            TEXT_COLOR,

            2
        )

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"RAW: {raw_prediction}",

            (20, 78),

            cv2.FONT_HERSHEY_DUPLEX,

            0.6,

            TEXT_COLOR,

            2
        )

        # ----------------------------------------------------
        # OPEN PROBABILITY
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"OPEN: {open_probability * 100:.1f}%",

            (20, 110),

            cv2.FONT_HERSHEY_DUPLEX,

            0.6,

            TEXT_COLOR,

            2
        )

        # ----------------------------------------------------
        # CLOSED PROBABILITY
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"CLOSED: {closed_probability * 100:.1f}%",

            (20, 142),

            cv2.FONT_HERSHEY_DUPLEX,

            0.6,

            TEXT_COLOR,

            2
        )

        # ----------------------------------------------------
        # CLOSED TIMER
        # ----------------------------------------------------

        if temporal_prediction == "CLOSED":

            cv2.putText(

                frame,

                f"CLOSED TIME: {closed_time:.2f}s",

                (20, 178),

                cv2.FONT_HERSHEY_DUPLEX,

                0.65,

                TEXT_COLOR,

                2
            )

        else:

            cv2.putText(

                frame,

                "CLOSED TIME: 0.00s",

                (20, 178),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                TEXT_COLOR,

                2
            )

        # ----------------------------------------------------
        # ALARM STATUS
        # ----------------------------------------------------

        if alarm_triggered:

            alarm_text = "ALARM ACTIVE"

        else:

            alarm_text = "ALARM ARMED"

        cv2.putText(

            frame,

            alarm_text,

            (20, 215),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            TEXT_COLOR,

            2
        )

        # ----------------------------------------------------
        # EAR
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"EAR: {feature_dict['ear']:.4f}",

            (20, 250),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            TEXT_COLOR,

            2
        )

    else:

        cv2.putText(

            frame,

            "NO FACE DETECTED",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            TEXT_COLOR,

            2
        )


    # ========================================================
    # CONTROL MESSAGE
    # ========================================================

    cv2.putText(

        frame,

        "ESC / Q = EXIT",

        (20, frame.shape[0] - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        TEXT_COLOR,

        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Eye Closure Alarm V2",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):

        print()
        print("Exit requested.")

        break


# ============================================================
# CLEANUP
# ============================================================

print()
print("Cleaning up...")

face_mesh.close()

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("EYE CLOSURE ALARM TEST COMPLETE")
print("=" * 70)

print()

print(
    f"Total frames : {total_frames}"
)

print(
    f"Face frames  : {face_frames}"
)

if total_frames > 0:

    print(
        f"Face detection rate : "
        f"{face_frames / total_frames * 100:.2f}%"
    )

print()

print("Model:")
print(MODEL_FILE)

print()

print("Temporal window:")
print(TEMPORAL_WINDOW)

print()

print("Closed alarm threshold:")
print(
    f"{CLOSED_DURATION:.1f} seconds"
)

print()

print("=" * 70)