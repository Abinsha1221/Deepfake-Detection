import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < 30:  # Get first 30 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, (128, 128))
            frame = frame / 255.0  # Normalize
            frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        if frames:
            return np.array(frames)
        return None
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def prepare_data():
    X = []  # Frames
    y = []  # Labels
    
    # Process real videos
    real_dir = 'dataset/real'
    for video_file in os.listdir(real_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            frames = process_video(os.path.join(real_dir, video_file))
            if frames is not None:
                X.extend(frames)
                y.extend([0] * len(frames))  # 0 for real
                print(f"Processed real video: {video_file}")

    # Process fake videos
    fake_dir = 'dataset/fake'
    for video_file in os.listdir(fake_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            frames = process_video(os.path.join(fake_dir, video_file))
            if frames is not None:
                X.extend(frames)
                y.extend([1] * len(frames))  # 1 for fake
                print(f"Processed fake video: {video_file}")

    return np.array(X), np.array(y)

def main():
    # Check if directories exist
    if not os.path.exists('dataset/real') or not os.path.exists('dataset/fake'):
        print("Please create 'dataset/real' and 'dataset/fake' directories and add videos")
        return
        
    print("Starting data preparation...")
    X, y = prepare_data()
    
    if len(X) == 0:
        print("No videos were processed successfully")
        return
        
    print(f"Total frames processed: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Creating and training model...")
    model = create_model()
    
    # Train the model
    history = model.fit(X_train, y_train,
                       epochs=10,
                       batch_size=32,
                       validation_data=(X_test, y_test))
    
    # Save the model
    model.save('model/deepfake_detection_model.h5')
    print("Model saved as 'deepfake_detection_model.h5'")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()

