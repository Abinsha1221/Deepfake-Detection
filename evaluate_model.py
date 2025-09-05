import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the model
model = tf.keras.models.load_model('model/deepfake_detection_model.h5')
print("✅ Model loaded successfully.")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Assuming EfficientNet input size
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames(video_path)
    if len(frames) == 0:
        return None
    preds = model.predict(frames, verbose=0)
    video_prediction = np.mean(preds)  # Average across frames
    return 1 if video_prediction >= 0.5 else 0

def evaluate_model():
    y_true = []
    y_pred = []

    real_dir = 'dataset/real'
    fake_dir = 'dataset/fake'

    # Real videos (label 0)
    for video_file in os.listdir(real_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            path = os.path.join(real_dir, video_file)
            pred = predict_video(path)
            if pred is not None:
                y_true.append(0)
                y_pred.append(pred)
                print(f"Processed REAL video: {video_file} → Prediction: {'Real' if pred == 0 else 'Fake'}")

    # Fake videos (label 1)
    for video_file in os.listdir(fake_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            path = os.path.join(fake_dir, video_file)
            pred = predict_video(path)
            if pred is not None:
                y_true.append(1)
                y_pred.append(pred)
                print(f"Processed FAKE video: {video_file} → Prediction: {'Fake' if pred == 1 else 'Real'}")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n📊 Evaluation Results:")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")

if __name__ == "__main__":
    evaluate_model()
