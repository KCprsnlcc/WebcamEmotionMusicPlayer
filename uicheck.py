import cv2
import matplotlib.pyplot as plt

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a frame from the camera
ret, frame = cap.read()

# Check if the frame was successfully captured
if not ret:
    print("Error: Could not capture frame.")
    exit()

# Convert the color space of the frame from BGR to RGB
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the original and converted frames using Matplotlib
plt.subplot(1, 2, 1)
plt.title('Original Frame (BGR)')
plt.imshow(frame)
plt.subplot(1, 2, 2)
plt.title('Converted Frame (RGB)')
plt.imshow(rgb_frame)
plt.show()

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
