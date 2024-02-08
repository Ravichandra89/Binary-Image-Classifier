import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('animal_injury_classifier.h5')

def predict_image(image):
    # Preprocess the image
    img = cv2.resize(image, (256, 256)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Return prediction
    return prediction[0][0]

def main():
    st.title('Animal Injury Classification')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        prediction = predict_image(image)
        if prediction > 0.5:
            st.write("Prediction: The animal is Injured")
        else:
            st.write("Prediction: The animal is Not Injured")

if __name__ == "__main__":
    main()
