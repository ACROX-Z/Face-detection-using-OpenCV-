import streamlit as st
import cv2
import dlib
import random
import time
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
from fer import FER
from textblob import TextBlob

# --- Configurations ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.20
MOUTH_AR_THRESH = 0.65

# --- Load Models ---
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

try:
    detector_fer = FER(mtcnn=True)
except:
    detector_fer = FER(mtcnn=False)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# --- Utility Functions ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def get_emoji(sentiment):
    if sentiment > 0:
        return "😀"
    elif sentiment < 0:
        return "😠"
    else:
        return "😐"

# --- Verification Function ---
def verify_face():
    blink_required = random.randint(3, 5)
    mouth_required = random.randint(2, 4)

    blink_count = 0
    mouth_count = 0
    blink_flag = False
    mouth_flag = False

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector_dlib(gray)

        if len(faces) > 1:
            cv2.putText(frame, "❌ More than one person detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            stframe.image(frame, channels="BGR")
            time.sleep(3)
            cap.release()
            return "❌ Verification failed: Multiple people detected."

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            if ear < EYE_AR_THRESH:
                if not blink_flag:
                    blink_flag = True
            else:
                if blink_flag:
                    blink_count += 1
                    blink_flag = False

            if mar > MOUTH_AR_THRESH:
                if not mouth_flag:
                    mouth_flag = True
            else:
                if mouth_flag:
                    mouth_count += 1
                    mouth_flag = False

            cv2.putText(frame, f"Blinks: {blink_count}/{blink_required}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Mouth Opens: {mouth_count}/{mouth_required}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            if blink_count >= blink_required and mouth_count >= mouth_required:
                cv2.putText(frame, "✅ Verification Successful!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                stframe.image(frame, channels="BGR")
                cap.release()
                return "✅ Realness Verification completed successfully!"

        stframe.image(frame, channels="BGR")

    cap.release()
    return "❌ Verification cancelled or failed."

# --- Expression Analysis Function ---
def analyze_expression():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    result_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector_fer.detect_emotions(rgb_frame)

        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            blob = TextBlob(top_emotion)
            sentiment = blob.sentiment.polarity
            emoji = get_emoji(sentiment)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{top_emotion} {emoji}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            result_text.markdown(f"### Expression: `{top_emotion}` {emoji}")

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Streamlit App UI ---
st.set_page_config(page_title="Face Realness & Expression App", layout="centered")
st.title("🧠 Realness Verification & Expression Detection")
st.markdown("Use your webcam to verify you're real and analyze your emotion.")

option = st.selectbox("Select Feature", ["Face Realness Verification", "Expression Analysis"])

if st.button("Start"):
    if option == "Face Realness Verification":
        result = verify_face()
        st.success(result) if "✅" in result else st.error(result)
    elif option == "Expression Analysis":
        analyze_expression()
