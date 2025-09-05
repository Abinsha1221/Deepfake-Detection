import tensorflow as tf
import numpy as np
import cv2
import os  # Add this import




class DeepfakeDetector:
    def __init__(self):
        self.model = self._load_model()    

    def _load_model(self):
        try:
            # Load the pre-trained model
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'model', 
                                    'deepfake_model.h5')
            model = tf.keras.models.load_model(model_path)
            print("Pre-trained model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_frame(self, frame, frame_size=(224, 224)):
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.keras.applications.efficientnet.preprocess_input(frame)
        return np.expand_dims(frame, axis=0)

    def predict_face(self, face):
        preprocessed_face = self.preprocess_frame(face)
        pred = self.model.predict(preprocessed_face)
        return pred[0][0]

    
    def analyze_video(self, video_path):
        threshold = 0.5
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(video_path)
        fake = 0
        real = 0 
    
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    prediction = self.predict_face(face)

                    if prediction > threshold:
                        fake += 1
                    else:
                        real += 1
                    
                # Remove visualization code for server use
                # label = 'FAKE' if prediction > threshold else 'REAL'
                # color = (0, 0, 255) if label == 'FAKE' else (0, 255, 0)
                # cv2.putText(frame, f'{label}: {prediction:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Remove visualization code for server use
            # cv2.imshow('Live Video', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        finally:
        # Always ensure resources are released
            cap.release()
        # cv2.destroyAllWindows() - Not needed if not displaying

        if fake > real:
            is_deepfake = True
        else:
            is_deepfake = False

        return {
            'is_deepfake': is_deepfake,
            'confidence': (fake/(fake+real)) * 100 if fake+real > 0 else 0,
            'message': 'Deepfake detected' if is_deepfake else 'No deepfake detected'
        }