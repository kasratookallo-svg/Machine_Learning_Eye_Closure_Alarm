import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from collections import deque



# CONFIGURATION

MODEL_FILE = "eye_random_forest_v2.pkl"

WINDOW_SIZE = 5

WINDOW_NAME = "Real-Time Temporal Eye Detection V2"

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



# MEDIAPIPE


mp_face_mesh = mp.solutions.face_mesh



# DISTANCE


def distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )



# FEATURE EXTRACTION
#
# IMPORTANT:
# USE NORMALIZED MEDIAPIPE COORDINATES
#
# DO NOT multiply x/y by image width/height.


def extract_features(landmarks):

    points = []

    for landmark in landmarks:

        points.append(
            (
                landmark.x,
                landmark.y
            )
        )



    # LEFT EYE LANDMARKS


    left_362 = points[362]
    left_385 = points[385]
    left_387 = points[387]
    left_263 = points[263]
    left_373 = points[373]
    left_380 = points[380]



    # RIGHT EYE LANDMARKS


    right_33 = points[33]
    right_160 = points[160]
    right_158 = points[158]
    right_133 = points[133]
    right_153 = points[153]
    right_144 = points[144]



    # EYE WIDTH


    left_eye_width = distance(
        left_362,
        left_263
    )

    right_eye_width = distance(
        right_33,
        right_133
    )



    # EYE HEIGHT


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



    # EAR


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



    # FEATURE DICTIONARY


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



# TEMPORAL MAJORITY VOTING


def get_stable_prediction(prediction_buffer):

    if len(prediction_buffer) == 0:

        return None


    closed_count = prediction_buffer.count("CLOSED")

    open_count = prediction_buffer.count("OPEN")


    if closed_count > open_count:

        return "CLOSED"

    else:

        return "OPEN"



# BUFFER DISPLAY


def get_buffer_text(prediction_buffer):

    result = []

    for state in prediction_buffer:

        if state == "OPEN":

            result.append("O")

        elif state == "CLOSED":

            result.append("C")

    return " ".join(result)



# START


print("=" * 70)
print("REAL-TIME TEMPORAL EYE DETECTION - RANDOM FOREST V2")
print("=" * 70)



# LOAD MODEL


print()
print("Loading model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")



# MODEL INFORMATION


print()
print("Classes:")
print(model.classes_)

print()
print("Trees:")
print(model.n_estimators)



# REAL-TIME PERFORMANCE


model.set_params(
    n_jobs=1

)

print()
print("Real-time n_jobs:")
print(model.n_jobs)



# TEMPORAL SETTINGS


prediction_buffer = deque(
    maxlen=WINDOW_SIZE
)

print()
print("Temporal window:")
print(WINDOW_SIZE)

print()
print("Temporal method:")
print("Majority Voting")



# OPEN WEBCAM


print()
print("Opening webcam...")

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Cannot open webcam."
    )



# MEDIAPIPE FACE MESH


face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)



# OPENCV WINDOW


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1000,
    700
)



# COUNTERS


total_frames = 0

face_frames = 0



# MAIN LOOP


while True:

    # READ FRAME

    ret, frame = cap.read()


    if not ret:

        print(
            "ERROR: Cannot read webcam frame."
        )

        break


    total_frames += 1


    # MIRROR IMAGE

    frame = cv2.flip(
        frame,
        1
    )


    # BGR -> RGB

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MEDIAPIPE

    results = face_mesh.process(
        rgb
    )


    # RESET FRAME VALUES

    raw_prediction = None

    stable_prediction = None

    open_probability = 0.0

    closed_probability = 0.0

    confidence = 0.0

    feature_dict = None


    # FACE DETECTED

    if results.multi_face_landmarks:

        face_frames += 1


        face_landmarks = (
            results.multi_face_landmarks[0]
        )


        # FEATURE EXTRACTION

        feature_dict = extract_features(
            face_landmarks.landmark
        )


        # MODEL INPUT

        X_live = pd.DataFrame(

            [[
                feature_dict[feature]
                for feature in FEATURES
            ]],

            columns=FEATURES
        )


        # RAW MODEL PREDICTION

        raw_prediction = model.predict(
            X_live
        )[0]


        # MODEL PROBABILITY

        probabilities = model.predict_proba(
            X_live
        )[0]


        class_probabilities = dict(
            zip(
                model.classes_,
                probabilities
            )
        )


        closed_probability = (
            class_probabilities.get(
                "CLOSED",
                0.0
            )
        )


        open_probability = (
            class_probabilities.get(
                "OPEN",
                0.0
            )
        )


        confidence = max(
            probabilities
        )


        # TEMPORAL BUFFER

        prediction_buffer.append(
            raw_prediction
        )



        # STABLE PREDICTION


        stable_prediction = get_stable_prediction(
            prediction_buffer
        )



    # DISPLAY

    if raw_prediction is not None:


        # RAW STATE

        cv2.putText(

            frame,

            f"RAW STATE: {raw_prediction}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2
        )


        # STABLE STATE

        cv2.putText(

            frame,

            f"STABLE STATE: {stable_prediction}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (0, 255, 0),

            2
        )


        # OPEN PROBABILITY

        cv2.putText(

            frame,

            f"OPEN: {open_probability * 100:.1f}%",

            (20, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


        # CLOSED PROBABILITY

        cv2.putText(

            frame,

            f"CLOSED: {closed_probability * 100:.1f}%",

            (20, 153),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


        # CONFIDENCE

        cv2.putText(

            frame,

            f"Confidence: {confidence * 100:.1f}%",

            (20, 186),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


        # EAR
        cv2.putText(

            frame,

            f"EAR: {feature_dict['ear']:.4f}",

            (20, 219),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


        # BUFFER

        buffer_text = get_buffer_text(
            prediction_buffer
        )


        cv2.putText(

            frame,

            f"BUFFER: {buffer_text}",

            (20, 252),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


    else:

        # NO FACE

        cv2.putText(

            frame,

            "NO FACE DETECTED",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.80,

            (0, 255, 0),

            2
        )


    # CONTROLS


    cv2.putText(

        frame,

        "Q / ESC = Quit",

        (20, frame.shape[0] - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 0),

        2
    )



    # SHOW FRAME


    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # KEYBOARD CONTROL

    key = cv2.waitKey(1) & 0xFF


    # Q OR ESC

    if key == ord("q"):

        print()
        print("Q pressed. Stopping...")

        break


    if key == ord("Q"):

        print()
        print("Q pressed. Stopping...")

        break


    if key == 27:

        print()
        print("ESC pressed. Stopping...")

        break


    # WINDOW CLOSED

    try:

        window_visible = cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        )

        if window_visible < 1:

            print()
            print("Window closed. Stopping...")

            break

    except:

        pass


# CLEANUP

print()
print("Cleaning up...")


face_mesh.close()

cap.release()

cv2.destroyAllWindows()


# FINAL REPORT


print()
print("=" * 70)
print("REAL-TIME TEMPORAL TEST COMPLETE")
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
print(WINDOW_SIZE)

print()

print("Temporal method:")
print("Majority Voting")

print()

print("=" * 70)
