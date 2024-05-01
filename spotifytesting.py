import cv2
import numpy as np
import spotipy
from keras.models import load_model
from spotipy.oauth2 import SpotifyClientCredentials
import os

# Load the pre-trained face detection and emotion recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('Model/modelv1.h5')  # Update model path

# Set up Spotify client credentials
client_id = '9adef84100074873b1fa65429882f5e0'
client_secret = '16f047e0463f4d4e9b914bc084598ddf'
sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


# Function to play music based on detected emotion
def play_music(emotion):
    # Map emotions to Spotify search queries
    emotion_queries = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'chill',  # Assuming neutral corresponds to a relaxed state
        'sad': 'sad',
        'surprise': 'surprise'
    }

    # Search for tracks based on the detected emotion
    search_query = emotion_queries.get(emotion, 'happy')  # Default to 'chill' if emotion not found
    search_results = sp.search(q=search_query, type='track', limit=1)
    tracks = search_results['tracks']['items']

    if tracks:
        # Play the first track
        track_uri = tracks[0]['uri']
        os.system(f'spotify play {track_uri}')


# Function to detect face and recognize emotion
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Predict emotion
        predicted_emotion = np.argmax(emotion_model.predict(face_roi))

        # Map predicted emotion index to emotion label
        emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        predicted_emotion_label = emotion_dict[predicted_emotion]

        # Play music based on detected emotion
        play_music(predicted_emotion_label)

        # Draw rectangle around the face and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame


# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_emotion(frame)

    # Display the webcam feed
    cv2.imshow('Face Emotion Music Player', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
