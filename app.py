import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ---- 1. CLASS NAMES FROM YOUR DATASET ----
class_names = ["Covid", "Normal", "Viral Pneumonia"]

# ---- 2. LOAD YOUR BEST MODEL ----
@st.cache_resource  # Won't reload every time
def load_my_model():
    model = load_model("best_vgg16_aug_model.h5")  # Name it as per your saving
    return model

model = load_my_model()

# ---- 3. IMAGE PREPROCESSING FUNCTION ----
def preprocess_image(img):
    # Convert to RGB (if uploaded as grayscale)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize to (224, 224) as per your model input
    img = img.resize((224, 224))
    # Convert to np.array and scale
    img = np.array(img).astype(np.float32) / 255.0  # Model expects scaled
    # Expand dims for batch (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img

# ---- 4. STREAMLIT APP LAYOUT ----
st.title("Chest X-ray Classifier: Covid | Normal | Viral Pneumonia")
st.write("Upload a chest X-ray image (.jpg, .jpeg, .png)")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_preprocessed = preprocess_image(image)

    # Predict
    prediction = model.predict(img_preprocessed)
    pred_class = np.argmax(prediction, axis=1)[0]
    pred_conf = prediction[0][pred_class]

    st.write(f"**Prediction:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {pred_conf:.2%}")

    st.bar_chart(dict(zip(class_names, prediction[0])))

    # If you want to show probabilities in detail
    st.write("**Class Probabilities:**")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]:.2%}")
