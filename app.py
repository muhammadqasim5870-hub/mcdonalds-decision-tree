import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "decision_tree_model.joblib")
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

st.set_page_config(
    page_title="McDonald's Menu Predictor",
    page_icon="🍔"
)

st.title("🍔 McDonald's Menu Tag Predictor")
st.write("Enter nutrition values to predict a menu tag.")

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

feature_cols = metadata["feature_cols"]

with st.form("prediction_form"):
    values = {}
    for feature in feature_cols:
        values[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=0.0,
            min_value=0.0,
            step=1.0
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame(
        [[values[col] for col in feature_cols]],
        columns=feature_cols
    )
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted tag: {prediction}")
    st.write("Input values:")
    st.dataframe(input_df, use_container_width=True)
