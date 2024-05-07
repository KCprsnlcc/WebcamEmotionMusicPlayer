import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import pygame
from keras.models import load_model
import os
import threading
from PIL import Image, ImageTk

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


class WebcamEmotionMusicPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Webcam Emotion Music Player")
        self.master.attributes("-fullscreen", True)

        self.camera = None
        self.camera_id = None
        self.camera_thread = None
        self.emotion_thread = None
        self.run_emotion_detection = False  # Flag to indicate whether emotion detection is running
        self.emotion_label_text = tk.StringVar(value="Emotion = None")

        self.create_widgets()

    def create_widgets(self):
        header_frame = tk.Frame(self.master, bg="#1e272e")
        header_frame.pack(side="top", fill="x")

        logo_img = Image.open("Resources/logo.png")
        logo_img = logo_img.resize((100, 100))
        self.logo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(header_frame, image=self.logo, bg="#1e272e")
        logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        title_description_frame = tk.Frame(header_frame, bg="#1e272e")
        title_description_frame.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        title_label = ttk.Label(
            title_description_frame,
            text="Webcam Emotion Music Player",
            font=("Helvetica", 16),
            foreground="white",
            background="#1e272e"
        )
        title_label.grid(row=0, column=0, sticky="w")

        description_label = ttk.Label(
            title_description_frame,
            text="Capture emotions through your webcam and enjoy music accordingly.",
            font=("Helvetica", 12),
            foreground="white",
            background="#1e272e"
        )
        description_label.grid(row=1, column=0, sticky="w")

        footer_frame = tk.Frame(self.master, bg="#1e272e")
        footer_frame.pack(side="bottom", fill="x")

        self.load_camera_btn = ttk.Button(
            footer_frame,
            text="Load Camera",
            command=self.load_camera,
            style="Flat.TButton"
        )
        self.load_camera_btn.pack(side="left", padx=10, pady=10)

        # New "Load Music" button
        self.load_music_btn = ttk.Button(
            footer_frame,
            text="Load Music",
            command=self.load_music,
            style="Flat.TButton"
        )
        self.load_music_btn.pack(side="left", padx=10, pady=10)

        # Existing buttons
        self.detect_emotion_btn = ttk.Button(
            footer_frame,
            text="Detect Emotion",
            command=self.detect_emotion,
            style="Amethyst.TButton"
        )
        self.detect_emotion_btn.pack(side="left", padx=10, pady=10)
        self.detect_emotion_btn.config(state="disabled")

        self.stop_btn = ttk.Button(
            footer_frame,
            text="Stop",
            command=self.stop_detection,
            style="Amethyst.TButton"
        )
        self.stop_btn.pack(side="left", padx=10, pady=10)
        self.stop_btn.config(state="disabled")

        self.fullscreen_btn = ttk.Button(
            footer_frame,
            text="Full Screen",
            command=self.toggle_fullscreen,
            style="IconButton.TButton"
        )
        self.fullscreen_btn.pack(side="left", padx=10, pady=10)

        self.exit_btn = ttk.Button(
            footer_frame,
            text="Exit",
            command=self.exit_application,
            style="Red.TButton"
        )
        self.exit_btn.pack(side="left", padx=10, pady=10)

        self.emotion_label = ttk.Label(
            footer_frame,
            textvariable=self.emotion_label_text,
            font=("Helvetica", 12),
            background="#1e272e",
            foreground="white",
            anchor="e"
        )
        self.emotion_label.pack(side="right", padx=10, pady=10)

        self.parent_panel = tk.Frame(self.master, bg="#FFFFFF")
        self.parent_panel.pack(fill="both", expand=True)

        self.camera_label = tk.Label(self.parent_panel, bg="#FFFFFF", borderwidth=0)
        self.camera_label.pack(fill="both", expand=True)

        self.master.style = ttk.Style()
        self.master.style.theme_use("clam")
        self.master.style.configure("Flat.TButton", background="#10ac84", foreground="white", font=("Helvetica", 12),
                                    borderwidth=0)
        self.master.style.configure("Amethyst.TButton", background="#9B59B6", foreground="white", font=("Helvetica", 12),
                                    borderwidth=0)
        self.master.style.configure("IconButton.TButton", background="#3498db", foreground="white",
                                    font=("Helvetica", 12), borderwidth=0)
        self.master.style.configure("Red.TButton", background="#ff6b6b", foreground="white", font=("Helvetica", 12),
                                    borderwidth=0)

    def load_music(self):
        # Implement loading music functionality here
        pass

    def load_camera(self):
        devices = self.get_camera_devices()
        if devices:
            dialog = CameraDeviceDialog(self.master, devices)
            window_width = 410
            window_height = 150
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.master.wait_window(dialog)

            if dialog.select_clicked:
                selected_device = dialog.selected_device.get()
                self.camera_id = int(selected_device.split()[1])
                self.camera = cv2.VideoCapture(self.camera_id)
                self.show_camera()
                if dialog.close_clicked:
                    self.hide_notification()
                else:
                    self.show_notification("Camera is now loaded successfully!")

    def get_camera_devices(self):
        devices = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(f"Camera {i}")
                cap.release()
        return devices

    def show_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.camera_id)

        ret, frame = self.camera.read()
        if ret:
            frame = self.resize_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.camera_label.img = img
            self.camera_label.config(image=img)
            self.camera_label.pack(fill="both", expand=True)
            self.detect_emotion_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.start()
            self.show_notification("Camera is now loaded successfully!")
        else:
            messagebox.showerror("Error", "Failed to load camera.")

    def show_notification(self, message):
        notification_window = tk.Toplevel(self.master)
        notification_window.title("Notification")
        notification_window.configure(background="#1e272e")
        window_width = 300
        window_height = 100
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        notification_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        notification_window.resizable(False, False)
        notification_window.attributes("-topmost", True)
        notification_window.overrideredirect(True)
        notification_frame = tk.Frame(notification_window, bg="#1e272e")
        notification_frame.pack(fill="both", expand=True)
        message_label = ttk.Label(
            notification_frame,
            text=message,
            font=("Helvetica", 12),
            foreground="white",
            background="#1e272e"
        )
        message_label.pack(padx=10, pady=10)
        close_button = ttk.Button(
            notification_frame,
            text="Close",
            command=notification_window.destroy,
            style="Flat.TButton"
        )
        close_button.pack(padx=10, pady=10)
        style = ttk.Style(notification_window)
        style.theme_use("clam")
        style.configure("Toplevel", background="#1e272e")
        style.configure("TLabel", background="#1e272e", foreground="white")
        style.configure("Flat.TButton", background="#10ac84", foreground="white", font=("Helvetica", 12), borderwidth=0)

    def resize_frame(self, frame):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        scale_x = screen_width / frame_width
        scale_y = screen_height / frame_height
        scale = min(scale_x, scale_y)
        frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
        return frame

    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            frame = self.resize_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.camera_label.img = img
            self.camera_label.config(image=img)
        self.master.after(10, self.update_camera)

    def detect_emotion(self):
        self.detect_emotion_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.run_emotion_detection = True  # Set flag to indicate emotion detection is running
        self.emotion_thread = threading.Thread(target=self.detect_emotion_process)
        self.emotion_thread.start()

    def detect_emotion_process(self):
        while self.run_emotion_detection:
            ret, frame = self.camera.read()
            if ret:
                frame = self.resize_frame(frame)
                frame, emotion = detect_emotion(frame)
                self.emotion_label_text.set(f"Emotion = {emotion.capitalize()}")

    def stop_detection(self):
        self.run_emotion_detection = False  # Set flag to stop emotion detection
        self.detect_emotion_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def toggle_fullscreen(self):
        self.master.attributes("-fullscreen", not self.master.attributes("-fullscreen"))

    def exit_application(self):
        self.run_emotion_detection = False  # Stop emotion detection before exiting
        if self.camera is not None:
            self.camera.release()
        self.master.destroy()

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

                return frame, predicted_emotion_label

    return frame, "None"

class CameraDeviceDialog(tk.Toplevel):
    def __init__(self, master, devices):
        super().__init__(master)
        self.title("Choose Camera Device")
        self.devices = devices
        self.close_clicked = False
        self.select_clicked = False

        self.selected_device = tk.StringVar()
        self.selected_device.set(self.devices[0])

        self.create_widgets()

    def create_widgets(self):
        self.selected_label = ttk.Label(
            self,
            textvariable=self.selected_device,
            font=("Helvetica", 12),
            background="#1e272e",
            foreground="white",
            anchor="center"
        )
        self.selected_label.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=10,
            pady=10,
            sticky="ew"
        )

        self.label = ttk.Label(
            self,
            text="Select Camera Device:",
            font=("Helvetica", 12),
            background="#1e272e"
        )
        self.label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.device_combobox = ttk.Combobox(
            self,
            textvariable=self.selected_device,
            values=self.devices,
            font=("Helvetica", 12),
            state="readonly"
        )
        self.device_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.close_button = ttk.Button(
            self,
            text="Close",
            command=self.close_and_exit,
            style="Red.TButton"
        )
        self.close_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.select_button = ttk.Button(
            self,
            text="Select",
            command=self.select_device,
            style="Flat.TButton"
        )
        self.select_button.grid(row=2, column=1, padx=10, pady=10, sticky="e")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("Toplevel", background="#1e272e")
        self.style.configure("TLabel", background="#1e272e", foreground="white")
        self.style.map("TCombobox", fieldbackground=[("readonly", "white")])
        self.style.configure("Flat.TButton", background="#10ac84", foreground="white", font=("Helvetica", 12), borderwidth=0)
        self.style.configure("Red.TButton", background="#ff6b6b", foreground="white", font=("Helvetica", 12), borderwidth=0)

        self.overrideredirect(True)

    def close_and_exit(self):
        self.close_clicked = True
        self.destroy()

    def select_device(self):
        self.select_clicked = True
        self.destroy()

    def get_camera_devices(self):
        devices = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(f"Camera {i}")
                cap.release()
        return devices



def main():
    root = tk.Tk()
    app = WebcamEmotionMusicPlayer(root)
    root.geometry("1366x768")
    root.mainloop()

if __name__ == "__main__":
    main()


