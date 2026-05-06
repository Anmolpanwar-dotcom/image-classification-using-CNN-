import json
import os
import shutil
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Pet Classifier AI", page_icon="🐾", layout="centered")

st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%; border-radius: 20px; height: 3em;
        background-color: #FF4B4B; color: white; font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def _make_keras3_config_tf215_compatible(value):
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value.get("config", {}).get("name", "float32")

        patched = {
            key: _make_keras3_config_tf215_compatible(item)
            for key, item in value.items()
        }

        if "batch_shape" in patched and "batch_input_shape" not in patched:
            patched["batch_input_shape"] = patched.pop("batch_shape")

        return patched

    if isinstance(value, list):
        return [_make_keras3_config_tf215_compatible(item) for item in value]

    return value


def _patch_h5_model_config(model_path: str) -> str:
    import h5py

    patched_path = os.path.join(tempfile.gettempdir(), "pet_model_tf215_compatible.h5")
    shutil.copyfile(model_path, patched_path)

    with h5py.File(patched_path, "r+") as h5_file:
        raw_config = h5_file.attrs.get("model_config")
        if raw_config is None:
            return patched_path

        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

        model_config = _make_keras3_config_tf215_compatible(json.loads(raw_config))
        h5_file.attrs["model_config"] = json.dumps(model_config)

    return patched_path


@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(
        repo_id="CodeWithAnmol/pet-classifier",
        filename="pet_model.h5",
    )

    patched_model_path = _patch_h5_model_config(model_path)
    return tf.keras.models.load_model(patched_model_path, compile=False)


st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image, and our AI will tell you if it's a Cat or a Dog!")

with st.sidebar:
    st.header("About Project")
    st.info("BCA Graduate (2022-2025) Portfolio Project. Built using CNN and TensorFlow.")
    st.write("---")
    st.markdown("### Tech Stack:")
    st.code("Python\nTensorFlow\nStreamlit\nCNN Architecture")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Analysis")
        if st.button("Predict"):
            with st.spinner("AI is thinking..."):
                try:
                    model = load_my_model()

                    img = image.resize((150, 150))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    prediction = model.predict(img_array, verbose=0)[0][0]

                    st.markdown(
                        '<div style="padding:20px;border-radius:15px;text-align:center;background:white;box-shadow:0 4px 6px rgba(0,0,0,0.1); color: black;">',
                        unsafe_allow_html=True,
                    )

                    if prediction > 0.5:
                        st.subheader("It's a DOG! 🐶")
                        st.write(f"Confidence: {float(prediction) * 100:.2f}%")
                    else:
                        st.subheader("It's a CAT! 🐱")
                        st.write(f"Confidence: {float(1 - prediction) * 100:.2f}%")

                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction mein error aaya: {e}")
else:
    st.warning("Please upload an image file to proceed.")

st.markdown("---")
st.caption("Developed by Anmol Panwar | Machine Learning Aspirant")
