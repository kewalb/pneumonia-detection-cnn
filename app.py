
import streamlit as st
import tensorflow as tf
import numpy as np
import pydicom
from PIL import Image


# Configuration
MODEL_PATH = (
    "best_pneumonia_model.keras"
)

IMAGE_SIZE = (224, 224)


# Load Model
@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# Streamlit Page
st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered"
)

st.title("Pneumonia Detection")

st.write(
    "Upload a chest X-ray DICOM (.dcm) image "
    "to predict whether pneumonia is present."
)


# Extract Image from DICOM
def extract_image(dicom):

    image = dicom.pixel_array.astype(
        np.float32
    )

    # Handle MONOCHROME1 images
    if getattr(
        dicom,
        "PhotometricInterpretation",
        ""
    ) == "MONOCHROME1":

        image = np.max(image) - image

    # Normalize pixel values to 0-255
    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:

        image = (
            (image - image_min)
            / (image_max - image_min)
            * 255.0
        )

    else:

        image = np.zeros_like(image)

    return image.astype(
        np.uint8
    )


# Preprocess DICOM for Model
def preprocess_dicom(dicom):

    image = extract_image(
        dicom
    )

    # Resize to 224 x 224
    image = Image.fromarray(
        image
    )

    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to NumPy
    image = np.array(
        image,
        dtype=np.float32
    )

    # Normalize to 0-1
    image = image / 255.0

    # Add grayscale channel
    # (224, 224) -> (224, 224, 1)
    image = np.expand_dims(
        image,
        axis=-1
    )

    # Add batch dimension
    # (224, 224, 1) -> (1, 224, 224, 1)
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# Upload DICOM
uploaded_file = st.file_uploader(
    "Upload Chest X-ray DICOM",
    type=["dcm"]
)


# Prediction

if uploaded_file is not None:

    try:

        # Read DICOM file
        # force=True allows files without DICM prefix
        dicom = pydicom.dcmread(
            uploaded_file,
            force=True
        )

        # Extract image
        display_image = extract_image(
            dicom
        )

        # Display image
        st.image(
            display_image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

        # Preprocess image
        image = preprocess_dicom(
            dicom
        )

        # Make prediction
        probability = model.predict(
            image,
            verbose=0
        )[0][0]

        # Classification
        if probability >= 0.5:

            prediction = "Pneumonia"

        else:

            prediction = "Normal"

        # Results
        st.subheader("Prediction")

        st.write(
            f"**Predicted Class:** {prediction}"
        )

        st.write(
            f"**Pneumonia Probability:** "
            f"{probability:.2%}"
        )

        st.progress(
            float(probability)
        )

    except Exception as e:

        st.error(
            f"Unable to process the DICOM file: {e}"
        )
