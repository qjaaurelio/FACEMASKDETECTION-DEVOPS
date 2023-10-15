import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('FacemaskModel')
    return model

model = load_model()

st.write("""
# Mask Detection System
""")

desired_size = (128, 128)  # Adjust to your model's input size

def preprocess_frame(frame):
    frame = cv2.resize(frame, desired_size)
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        preprocessed_frame = preprocess_frame(frame)
        predictions = model.predict(preprocessed_frame)

        if predictions[0][0] < 0.5:
            label = "With Mask"
        else:
            label = "Without Mask"

        # Overlay the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if webrtc_ctx.video_receiver:
    webrtc_ctx.video_receiver.on_data(data)
