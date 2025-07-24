import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Define class names in correct order (match train_data.class_indices)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# App title
st.title("🧠 Brain Tumor Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an MRI brain scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    image = image.resize((128, 128))  # Match model input
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape((1, 128, 128, 3))  # Add batch dim

    # Predict
    prediction = model.predict(img_array)[0]  # Get first row
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Show result
    st.markdown(f"### 🧠 Prediction: **{predicted_class.upper()}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

# Optional: Add a clear button
if st.button("Clear"):
    st.experimental_rerun()
