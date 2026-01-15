from unittest import result
import requests
import streamlit as st

st.set_page_config(page_title="Indonesian Food Classifier")

st.title("🇮🇩 Indonesian Food Image Classification")

model_ml = [
    "logreg",
    "rf"
]

uploaded = st.file_uploader("Upload food image", type=["jpg", "png"])
model_name = st.selectbox("Select model", model_ml)

API_URL = "http://localhost:8000/predict"

if uploaded:
    files = {"file": uploaded.getvalue()}
    params = {"model_name": model_name.lower().replace(" ", "")}

    r = requests.post(API_URL, files=files, params=params)
    result = r.json()

    st.metric("Prediction", result["predicted_class"])
    st.metric("Confidence", f"{result['probability']:.2f}")