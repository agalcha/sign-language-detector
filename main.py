# ==============================
# ASL Detection Project (Colab)
# ==============================

# --- Imports ---
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from google.colab import output
from google.colab.output import eval_js
from IPython.display import display, Javascript, clear_output
from base64 import b64decode
import ipywidgets as widgets

# --- Global Flags ---
global_stop_button_pressed = False
global_is_detecting = False

# --- Paths & Config ---
DATA_PATH = "asl_data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

SIGNS = ['A', 'B', 'C', 'I_Love_You']
NUM_SAMPLES_PER_SIGN = 10  # samples per sign

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

print("✅ Setup Complete! Ready to go.")

# --- Webcam Frame Capture ---
def get_video_frame(quality=0.8):
    js = Javascript('''
      async function captureFrame() {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = '📸 Capture Frame';
        div.appendChild(capture);

        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});

        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        div.remove();
        return canvas.toDataURL('image/jpeg', %f);
      }
      captureFrame();
    ''' % quality)

    display(js)
    data = eval_js('captureFrame()')
    binary = b64decode(data.split(',')[1])
    np_arr = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


# --- Data Collection ---
def collect_data():
    global global_stop_button_pressed
    print("📷 Starting Data Collection...")

    for sign in SIGNS:
        if global_stop_button_pressed:
            print("🛑 Data collection stopped.")
            global_stop_button_pressed = False
            break

        sign_path = os.path.join(DATA_PATH, sign)
        os.makedirs(sign_path, exist_ok=True)

        clear_output(wait=True)
        print(f"\n👉 Collect data for sign: '{sign}'")
        print(f"Please perform the sign. Need {NUM_SAMPLES_PER_SIGN} samples.")

        collected_count = 0
        while collected_count < NUM_SAMPLES_PER_SIGN:
            frame = get_video_frame()
            frame = cv2.flip(frame, 1)

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x - wrist.x)
                        landmarks.append(lm.y - wrist.y)

                    if len(landmarks) == 42:
                        file_path = os.path.join(sign_path, f"{collected_count}.csv")
                        with open(file_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(landmarks)
                        collected_count += 1

                        clear_output(wait=True)
                        print(f"✅ Captured {collected_count}/{NUM_SAMPLES_PER_SIGN} for '{sign}'")
                    else:
                        print("⚠️ Invalid landmark length. Skipped.")
            else:
                print("❌ No hand detected. Try again.")

    clear_output(wait=True)
    print("🎉 Data Collection Finished!")


# --- Training ---
def train_model():
    print("📊 Training KNN Model...")

    X_data, y_data = [], []
    for sign_folder in os.listdir(DATA_PATH):
        sign_path = os.path.join(DATA_PATH, sign_folder)
        if not os.path.isdir(sign_path):
            continue

        for sample_file in os.listdir(sign_path):
            file_path = os.path.join(sign_path, sample_file)
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                landmarks = list(reader)[0]

                if len(landmarks) == 42:
                    X_data.append([float(v) for v in landmarks])
                    y_data.append(sign_folder)
                else:
                    print(f"⚠️ Skipped {file_path}, invalid length {len(landmarks)}")

    if not X_data:
        print("❌ No valid data. Collect data first.")
        return None, None

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_data, y_data)

    print(f"✅ Training Done! Samples: {len(X_data)}")
    return knn_model, y_data


# --- Detection ---
def start_detection(model):
    global global_is_detecting, global_stop_button_pressed
    if model is None:
        print("⚠️ Train model first.")
        return

    global_is_detecting = True
    global_stop_button_pressed = False

    clear_output(wait=True)
    print("🔍 Detection Started")
    print("👉 Press 'Capture Gesture' to classify.")
    print("👉 Press 'Stop Detection' to exit.")
    display(widgets.VBox([capture_button, stop_button]))


def on_capture_button_clicked(b):
    global trained_knn_model, global_is_detecting, global_stop_button_pressed

    if not global_is_detecting or global_stop_button_pressed:
        return

    frame = get_video_frame()
    frame = cv2.flip(frame, 1)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    clear_output(wait=True)
    display(widgets.VBox([capture_button, stop_button]))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - wrist.x)
                landmarks.append(lm.y - wrist.y)

            if len(landmarks) == 42:
                prediction_data = np.array(landmarks).reshape(1, -1)
                predicted_sign = trained_knn_model.predict(prediction_data)[0]
                confidence = np.max(trained_knn_model.predict_proba(prediction_data)) * 100
                print(f"👉 Prediction: {predicted_sign} ({confidence:.1f}%)")
            else:
                print("⚠️ Invalid landmark length.")
    else:
        print("❌ No hand detected.")


def on_stop_button_clicked(b):
    global global_stop_button_pressed, global_is_detecting
    global_stop_button_pressed = True
    global_is_detecting = False
    clear_output(wait=True)
    print("🛑 Detection Stopped.")
    display(ui_buttons_box)


# --- Buttons ---
data_collect_button = widgets.Button(description="Start Data Collection")
train_model_button = widgets.Button(description="Train Model")
start_detection_button = widgets.Button(description="Start Detection")
stop_button = widgets.Button(description="Stop Detection", button_style='danger')
capture_button = widgets.Button(description="Capture Gesture")

# --- Events ---
trained_knn_model, y_labels = None, None

def on_data_collect_button_clicked(b):
    global trained_knn_model, y_labels
    clear_output(wait=True)
    collect_data()
    trained_knn_model, y_labels = train_model()
    display(ui_buttons_box)

def on_train_model_button_clicked(b):
    global trained_knn_model, y_labels
    clear_output(wait=True)
    trained_knn_model, y_labels = train_model()
    display(ui_buttons_box)

def on_start_detection_button_clicked(b):
    clear_output(wait=True)
    start_detection(trained_knn_model)

data_collect_button.on_click(on_data_collect_button_clicked)
train_model_button.on_click(on_train_model_button_clicked)
start_detection_button.on_click(on_start_detection_button_clicked)
stop_button.on_click(on_stop_button_clicked)
capture_button.on_click(on_capture_button_clicked)

# --- UI ---
ui_buttons_box = widgets.VBox([
    widgets.Label("✋ ASL Detection Control Panel"),
    data_collect_button,
    train_model_button,
    start_detection_button
])

print("🎯 Ready! Use the buttons below:")
display(ui_buttons_box)
