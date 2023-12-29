import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/Image.h5')

# Streamlit app
st.title("🌟 Image Classification: Sad or Happy 🌟")

# Header and Observations
st.header("Welcome to the Image Classification App!")
st.write(
    "Upload an image, and I will predict whether the person in the image looks sad or happy."
)

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(256, 256))
    resized_image = image.img_to_array(img)
    input_data = np.expand_dims(resized_image / 255, 0)

    # Make predictions
    prediction = model.predict(input_data)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Stylish prediction result
    st.markdown("<h2 style='text-align: center;'>Prediction Result</h2>", unsafe_allow_html=True)
    prediction_text = "Sad" if prediction > 0.5 else "Happy"
    st.markdown(f"<h3 style='text-align: center; color: {'#3498db' if prediction > 0.5 else '#27ae60'};'>{prediction_text}</h3>", unsafe_allow_html=True)

    # Observations
    st.markdown("## Observations:")
    st.write(
        "- The model has been trained on a dataset of images labeled as Sad or Happy."
    )
    st.write(
        "- The predictions are made based on the facial expressions in the uploaded image."
    )
    st.write(
        "- The model output is a probability, and a threshold of 0.5 is used to classify as Sad or Happy."
    )
