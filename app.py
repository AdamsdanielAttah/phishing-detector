import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="centered",
)

MODEL_PATH = BASE_DIR / "phishing_rf_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"


@st.cache_resource
def load_model():
    """Load the trained model and the exact feature order used during training."""
    model = joblib.load(MODEL_PATH)
    with FEATURE_PATH.open(encoding="utf-8") as f:
        columns = json.load(f)
    return model, columns


try:
    model, FEATURE_COLUMNS = load_model()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.stop()

GROUPS = {
    "Address Bar Based Features": [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain",
        "SSLfinal_State", "Domain_registeration_length", "Favicon", "port",
        "HTTPS_token",
    ],
    "Abnormal-Based Features": [
        "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
        "Submitting_to_email", "Abnormal_URL",
    ],
    "HTML & JavaScript Based Features": [
        "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe",
    ],
    "Domain Based Features": [
        "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
        "Google_Index", "Links_pointing_to_page", "Statistical_report",
    ],
}

# Keep the UI and the model contract synchronized.
missing = [feature for group in GROUPS.values() for feature in group if feature not in FEATURE_COLUMNS]
if missing or len(FEATURE_COLUMNS) != 30:
    st.error("Feature configuration is invalid. Expected the 30 UCI phishing features.")
    if missing:
        st.write("Missing:", missing)
    st.stop()

st.title("🛡️ Phishing Website Detector")
st.caption("Project #23 · Random Forest Machine Learning Classifier")

st.info(
    "This project classifies a website using 30 pre-extracted phishing-detection "
    "features from the UCI Phishing Websites dataset. It is a feature-based ML "
    "classifier; it does not automatically visit or scan a URL."
)

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** Random Forest")
    st.write("**Features:** 30")
    st.write("**Dataset:** UCI Phishing Websites")
    st.write("**Classes:** Legitimate / Phishing")
    st.caption(
        "The model probability shown after prediction is the Random Forest's "
        "predicted class probability; it is not a guarantee of real-world safety."
    )

st.divider()

# ---------- Example presets ----------
example_legit = {feature: 1 for feature in FEATURE_COLUMNS}
example_phish = {feature: -1 for feature in FEATURE_COLUMNS}
example_neutral = {feature: 0 for feature in FEATURE_COLUMNS}

if "feature_values" not in st.session_state:
    st.session_state.feature_values = dict(example_neutral)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Legitimate example", use_container_width=True):
        st.session_state.feature_values = dict(example_legit)
        st.rerun()

with col2:
    if st.button("🚩 Phishing example", use_container_width=True):
        st.session_state.feature_values = dict(example_phish)
        st.rerun()

with col3:
    if st.button("↩️ Reset", use_container_width=True):
        st.session_state.feature_values = dict(example_neutral)
        st.rerun()

st.divider()
st.subheader("Website Feature Values")
st.caption(
    "Use the dataset encoding: **1 = legitimate indicator, 0 = suspicious/neutral, "
    "-1 = phishing indicator**. These values represent features extracted from a "
    "website; they are not ordinary user ratings."
)

for group_name, features in GROUPS.items():
    with st.expander(group_name, expanded=(group_name == "Address Bar Based Features")):
        cols = st.columns(3)
        for index, feature in enumerate(features):
            with cols[index % 3]:
                st.session_state.feature_values[feature] = st.select_slider(
                    feature,
                    options=[-1, 0, 1],
                    value=st.session_state.feature_values.get(feature, 0),
                    key=f"slider_{feature}",
                )

st.divider()

if st.button("🔍 Classify Features", type="primary", use_container_width=True):
    try:
        input_vector = np.array(
            [[st.session_state.feature_values[feature] for feature in FEATURE_COLUMNS]],
            dtype=int,
        )

        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        classes = list(model.classes_)

        if prediction not in classes:
            raise ValueError(f"Unexpected model class: {prediction}")

        predicted_probability = probabilities[classes.index(prediction)] * 100

        if prediction == 1:
            st.success(
                f"### ✅ Classified as Legitimate\n"
                f"**Model probability: {predicted_probability:.1f}%**"
            )
        elif prediction == -1:
            st.error(
                f"### 🚩 Classified as Phishing\n"
                f"**Model probability: {predicted_probability:.1f}%**"
            )
        else:
            st.warning(f"Unknown class returned by the model: {prediction}")

        with st.expander("See probability breakdown"):
            for class_value, probability in zip(classes, probabilities):
                label = "Legitimate" if class_value == 1 else "Phishing"
                st.write(f"**{label}:** {probability * 100:.2f}%")

    except Exception as exc:
        st.error("Prediction failed.")
        st.code(str(exc))

st.divider()
st.caption(
    "Random Forest model · 30 UCI phishing features · "
    "For educational/demo use; a real deployment should add automated feature extraction, "
    "URL validation, monitoring, and independent security testing."
)
