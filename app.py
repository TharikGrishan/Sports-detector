# ==============================================================================
# Streamlit Web Application for Image Classification
# This app uses a pre-trained MobileNetV2 model to classify sports images.
# ==============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Sports Classification App",
    page_icon="🏀",
    layout="centered"
)

# ------------------------------------------------------------------------------
# 2. Main Title and Description
# ------------------------------------------------------------------------------
st.title("⚽️ Sports Image Classifier")
st.write("Upload an image of a sport, and the model will predict what it is!")
st.write("---")

# ------------------------------------------------------------------------------
# 3. Load the Trained Model and Class Names
# ------------------------------------------------------------------------------

# Define the model and class name file paths.
MODEL_PATH = 'sports_classifier_mobilenet.keras'
CLASSES_PATH = 'class_names.txt'
IMAGE_SIZE = (224, 224)

# Use st.cache_resource to load the model only once.
# This prevents the model from reloading every time the app updates.
@st.cache_resource
def load_model_and_classes():
    """Loads the pre-trained Keras model and the class names."""
    try:
        # Load the model
        model = keras.models.load_model(MODEL_PATH)
        
        # Load the class names
        with open(CLASSES_PATH, 'r') as f:
            class_names = [line.strip() for line in f]
            
        return model, class_names
    except FileNotFoundError:
        st.error(f"Could not find the model file '{MODEL_PATH}' or class names file '{CLASSES_PATH}'.")
        st.info("Please make sure you have downloaded them from Google Colab and placed them in the same directory as this script.")
        return None, None

# Load the resources at the start
model, class_names = load_model_and_classes()

# ------------------------------------------------------------------------------
# 4. Image Upload and Prediction Logic
# ------------------------------------------------------------------------------

if model is not None and class_names is not None:
    # Create a file uploader widget
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Process the image for the model
        with st.spinner('Making a prediction...'):
            # Resize the image to match the model's input size
            image = image.resize(IMAGE_SIZE)
            
            # Convert the image to a NumPy array and add a batch dimension
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)
            
            # Normalize the pixel values (rescale from 0-255 to 0-1)
            image_array = image_array / 255.0

            # Make the prediction
            predictions = model.predict(image_array)
            
            # Get the predicted class and confidence
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index].replace('_', ' ').title()
            
            # Convert the NumPy float32 to a standard Python float
            confidence = float(np.max(predictions)) * 100
        
        # Display the results
        st.success("Classification Complete!")
        st.write(f"### The model predicts: **{predicted_class}**")
        st.progress(confidence / 100)
        st.write(f"**Confidence:** {confidence:.2f}%")
