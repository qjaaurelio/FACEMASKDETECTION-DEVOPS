import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('FacemaskModel')
  return model
model=load_model()
st.write("""
# Mask Detection System"""
)


desired_size = (128, 128)  # Adjust to your model's input size

def preprocess_frame(frame):
    frame = cv2.resize(frame, desired_size)
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame = preprocess_frame(frame)

    # Run inference with preprocessed_frame
    predictions = model.predict(preprocessed_frame)

    # Post-process the predictions
    if predictions[0][0] < 0.5:  # You can adjust this threshold
        label = "With Mask"
    else:
        label = "Without Mask"

    # Overlay the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the result
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
