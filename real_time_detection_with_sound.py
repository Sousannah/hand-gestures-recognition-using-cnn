import pyttsx3
import cv2
import numpy as np
import tensorflow as tf
import os

# Load the saved model
model = tf.keras.models.load_model('model01_cnn.h5')

labels = ['a', 'b', 'c', 'f', 'k', 'y']

# Initialize the video source (0 for internal camera)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Couldn't capture frame")
        break

    # Draw a rectangle for ROI
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5)
    roi = frame[100:300, 100:300]  # Region of interest

    # Resize the ROI to match the model's input shape
    img = cv2.resize(roi, (50, 50))
    img = img / 255.0  # Normalize pixel values

    # Make prediction about the current frame
    prediction = model.predict(img.reshape(1, 50, 50, 3))
    char_index = np.argmax(prediction)
    predicted_char = labels[char_index]

    confidence = round(prediction[0, char_index] * 100, 1)
    predicted_char = labels[char_index]

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2

    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(predicted_char)
    engine.runAndWait()

    msg = f"{predicted_char}, Conf: {confidence}%"
    cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)

    # Show the frame
    cv2.imshow('frame', frame)

    # Close the camera when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()