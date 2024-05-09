import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import pygame
from keras.models import load_model
import os
import random
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
        'angry': 'Angry',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        'surprise': 'Surprise'
    }

    if emotion in music_files:
        folder_name = music_files[emotion]
        music_path = os.path.join(music_folder, folder_name)
        music_list = os.listdir(music_path)

        if music_list:
            # Select a random music file from the folder
            music_file = random.choice(music_list)
            music_file_path = os.path.join(music_path, music_file)

            # Check if a music is currently playing
            if current_music is not None and current_music != music_file_path:
                pygame.mixer.music.stop()  # Stop currently playing music

            if current_music != music_file_path:
                pygame.mixer.music.load(music_file_path)
                pygame.mixer.music.play(-1)  # Loop the music indefinitely
                current_music = music_file_path  # Update currently playing music path


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
        # Create and configure the dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Load Music")
        dialog.configure(background="#1e272e")
        dialog.attributes("-topmost", True)

        # Remove the Windows buttons (minimize, maximize, close)
        dialog.overrideredirect(True)

        # Fix window size and make it non-resizable
        window_width = 940
        window_height = 600
        dialog.geometry(f"{window_width}x{window_height}")
        dialog.resizable(False, False)

        # Calculate the position to center the dialog window
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        dialog.geometry(f"+{x}+{y}")

        # Create a frame to hold the header (emotion category)
        header_frame = tk.Frame(dialog, bg="#1e272e")
        header_frame.pack(side="top", fill="x", padx=10, pady=10)

        # Create a frame to hold the music files for the selected emotion
        music_frame = tk.Frame(dialog, bg="#1e272e")
        music_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a scrollbar for the music frame
        scrollbar = ttk.Scrollbar(music_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Create a canvas to hold the music labels
        canvas = tk.Canvas(music_frame, bg="#1e272e", yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)

        # Configure the scrollbar to scroll the canvas
        scrollbar.config(command=canvas.yview)

        # Create a frame inside the canvas to hold the music labels
        music_container = tk.Frame(canvas, bg="#1e272e")
        canvas.create_window((0, 0), window=music_container, anchor="nw")

        # Function to load music files for a specific emotion category
        def load_emotion_music(emotion):
            # Clear the music frame
            for widget in music_container.winfo_children():
                widget.destroy()

            # Fetch music files in the emotion folder
            music_folder = os.path.join("Music", emotion.lower())
            music_files = [file for file in os.listdir(music_folder) if file.endswith(".mp3")]

            # Function to play the selected music file
            def play_music_file(music_file):
                music_file_path = os.path.join(music_folder, music_file)
                pygame.mixer.music.load(music_file_path)
                pygame.mixer.music.play(-1)  # Loop the music indefinitely

            # Function to delete the selected music file
            def delete_music_file(music_file, play_button):
                music_file_path = os.path.join(music_folder, music_file)
                if current_music == music_file_path:
                    pygame.mixer.music.stop()  # Stop the music if it's currently playing
                os.remove(music_file_path)
                play_button.config(state="disabled")  # Disable the play button after deleting the file
                # Reload the music files after deletion
                load_emotion_music(emotion)

            # Display music files with play, stop, and delete buttons
            for music_file in music_files:
                label_frame = tk.Frame(music_container, bg="#1e272e")
                label_frame.pack(fill="x")

                # Music title label
                label = ttk.Label(label_frame, text=music_file, font=("Helvetica", 12), background="#1e272e",
                                  foreground="white")
                label.pack(side="left", padx=(10, 5), pady=5)

                # Play button
                play_button = ttk.Button(label_frame, text="Play",
                                         command=lambda file=music_file: play_music_file(file), style="Flat.TButton")
                play_button.pack(side="right", padx=(5, 5), pady=5)

                # Stop button
                stop_button = ttk.Button(label_frame, text="Stop",
                                         command=lambda: pygame.mixer.music.stop(), style="Flat.TButton")
                stop_button.pack(side="right", padx=(5, 5), pady=5)

                # Delete button
                delete_button = ttk.Button(label_frame, text="Delete",
                                           command=lambda file=music_file, play_button=play_button: delete_music_file(
                                               file, play_button), style="Red.TButton")
                delete_button.pack(side="right", padx=(5, 5), pady=5)

            # Update the canvas scroll region
            canvas.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))

        # Create buttons for each emotion category
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        for emotion in emotions:
            ttk.Button(header_frame, text=emotion, command=lambda emo=emotion: load_emotion_music(emo),
                       style="Flat.TButton").pack(side="left", padx=10)

        # Function to close the dialog window
        def close_dialog():
            dialog.destroy()

        # Create a frame to hold the footer (close button)
        footer_frame = tk.Frame(dialog, bg="#1e272e")
        footer_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Add "Upload Music" button
        upload_button = ttk.Button(footer_frame, text="Upload Music", command=self.upload_music, style="Flat.TButton")
        upload_button.pack(side="left", padx=10, pady=10)

        # Add a label for uploading music files
        upload_label = ttk.Label(footer_frame, text="Upload your music file (.mp3)", background="#1e272e",
                                 foreground="white")
        upload_label.pack(side="left", padx=10)

        # Add a close button to close the dialog window
        close_button = ttk.Button(footer_frame, text="Close", command=close_dialog, style="Red.TButton")
        close_button.pack(side="right", padx=10)

        dialog.mainloop()

    def upload_music(self):
        # Implement the functionality to upload music here
        pass

    def animate_title(self, label):
        # Function to animate the music title label
        text = label.cget("text")
        while True:
            text = text[1:] + text[0]
            label.config(text=text)
            label.update()
            label.after(300)

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