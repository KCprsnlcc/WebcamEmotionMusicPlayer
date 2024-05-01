import cv2
import numpy as np
import pygame
from keras.models import load_model
import os
import threading

# Load the pre-trained face detection and emotion recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('Model/modelv1.h5')  # Update model path

# Initialize Pygame mixer
pygame.mixer.init()

# Variable to store the currently playing music path
current_music = None

# Function to play music based on emotion
def play_music(emotion):
    global current_music

    music_folder = "Music"
    music_files = {
        'angry': 'angry_music.mp3',
        'disgust': 'disgust_music.mp3',
        'fear': 'fear_music.mp3',
        'happy': 'happy_music.mp3',
        'neutral': 'neutral_music.mp3',
        'sad': 'sad_music.mp3',
        'surprise': 'surprise_music.mp3'
    }

    if emotion in music_files:
        music_path = os.path.join(music_folder, music_files[emotion])

        # Check if a music is currently playing
        if current_music is not None and current_music != music_path:
            pygame.mixer.music.stop()  # Stop currently playing music

        if current_music != music_path:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.play(-1)  # Loop the music indefinitely
            current_music = music_path  # Update currently playing music path

# Function to detect face and recognize emotion
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Ensure only one face is detected
    if len(faces) == 1:
        x, y, w, h = faces[0]  # Get coordinates of the detected face

        # Calculate aspect ratio
        aspect_ratio = float(w) / h

        # Ensure the face is frontal based on aspect ratio
        if aspect_ratio > 0.5 and aspect_ratio < 2.0:  # Adjust this range based on your specific requirements
            # Ensure the face is of sufficient size
            if w > 100 and h > 100:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = np.expand_dims(face_roi, axis=-1)

                # Predict emotion
                predicted_emotion = np.argmax(emotion_model.predict(face_roi))

                # Map predicted emotion index to emotion label
                emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
                predicted_emotion_label = emotion_dict[predicted_emotion]

                # Play music based on emotion
                play_music(predicted_emotion_label)

                # Draw rectangle around the face and display the emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Open webcam
cap = cv2.VideoCapture(0)

# Function to continuously read frames from the webcam
def webcam_thread():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_emotion(frame)

        # Display the webcam feed
        cv2.imshow('Face Emotion Music Player', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the webcam thread
webcam_thread = threading.Thread(target=webcam_thread)
webcam_thread.start()

# Keep the main thread running to handle music playback
while True:
    pass

# Release resources
cap.release()
cv2.destroyAllWindows()
