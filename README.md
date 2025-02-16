# BE KIND, WE'RE STILL IN BETA 🚧
![Beta](https://img.shields.io/badge/status-beta-yellow)
![Python](https://img.shields.io/badge/python-3.x-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Welcome to our Emotion Detection and Music Player application! 🎵 This application leverages computer vision and deep learning to detect emotions and play corresponding music.

## Requirements 📋

### System Requirements:
- **Python 3.x** installed on your system.

### Python Libraries:
- **tkinter**: For GUI development.
- **opencv-python**: For computer vision tasks.
- **numpy**: For numerical computations.
- **pygame**: For playing music.
- **keras**: For loading a pre-trained emotion recognition model.
- **PIL**: Python Imaging Library for image processing.

### Additional Libraries for Music Management:
- **os**: For file system operations such as copying and deleting music files.
- **threading**: For handling concurrent execution.

## How to Use 🚀

1. **Run the Application**:
   - Open a terminal or command prompt.
   - Navigate to the directory where the Python script is saved.
   - Run the script using the command: 
     ```sh
     python <main>.py
     ```

2. **Load Camera**:
   - Click on the `Load Camera` button to initialize the webcam.
   - If multiple cameras are available, select the desired one from the drop-down menu.

3. **Detect Emotion**:
   - After loading the camera, click on the `Detect Emotion` button to start the emotion detection process.
   - The application will continuously analyze the webcam feed to detect faces and emotions.

4. **Enjoy Music**:
   - Based on the detected emotion, the application will play corresponding music.
   - Each emotion has its predefined music file stored in the `Music` folder.

5. **Load Music**:
   - Click on the `Load Music` button to load music files corresponding to different emotions.
   - Ensure that the music files are correctly labeled and stored in the `Music` folder.

6. **Upload Music**:
   - Click on the `Upload Music` button to add new music files to the `Music` folder.
   - Select the music files from your system's file browser and click `Open` to upload them.

7. **Delete Music**:
   - Select the music file you want to delete from the list of loaded music files.
   - Click on the `Delete Music` button to remove the selected music file from the `Music` folder.

8. **Stop Detection**:
   - Click on the `Stop` button to pause the emotion detection process.

9. **Toggle Full Screen**:
   - Click on the `Full Screen` button to switch between full-screen and windowed mode.

10. **Exit Application**:
    - Click on the `Exit` button to close the application.
    - Ensure to stop emotion detection before exiting to release the camera resources.

## Notes 📓
- Ensure that the required emotion detection model (`modelv1.h5`) is present in the specified location.
- Music files corresponding to different emotions should be stored in the `Music` folder.
- The application uses pre-trained Haar cascade for face detection and a CNN model for emotion recognition.
- Adjust the aspect ratio and size thresholds in the `detect_emotion` function based on your requirements.
- You can customize the music files and their associations with different emotions in the `play_music` function.

## Libraries Used 📚

- **tkinter**: For GUI development.
- **cv2**: OpenCV library for computer vision tasks.
- **numpy**: For numerical computations.
- **pygame**: For playing music.
- **keras.models**: For loading a pre-trained emotion recognition model.
- **os**: For operating system operations.
- **threading**: For handling concurrent execution.
- **PIL**: Python Imaging Library for image processing.

## Functions and Classes 📄

1. **`load_music()`**: 
    - Function to load music files corresponding to different emotions from the `Music` folder.
    - This function can be called when the application starts or when the `Load Music` button is clicked.

2. **`upload_music()`**:
    - Function to upload new music files to the `Music` folder.
    - This function can be triggered when the `Upload Music` button is clicked.
    - It should open a file dialog to allow the user to select one or more music files from their system.
    - Once selected, the function should copy the chosen music files to the `Music` folder.

3. **`delete_music(file_name)`**:
    - Function to delete a specific music file from the `Music` folder.
    - This function can be called when the `Delete Music` button is clicked and a music file is selected from the list.
    - It should remove the selected music file from the `Music` folder.

4. **`MusicController`**:
    - Class to handle music-related operations such as loading, uploading, and deleting music files.
    - It can have methods like `load_music`, `upload_music`, and `delete_music`.

5. **`GUI`** (or any main application class):
    - Modify the existing GUI class or create a new one to include buttons for loading, uploading, and deleting music files.
    - Bind these buttons to the corresponding functions in the `MusicController`.

---

Thank you for trying our application! Your feedback is valuable to us. 🧡