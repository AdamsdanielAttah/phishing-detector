import streamlit as st
import joblib
import json
import numpy as np

st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️", layout="centered")

# ---------- Load the real trained model ----------
@st.cache_resource
def load_model():
    model = joblib.load("phishing_rf_model.pkl")
    with open("feature_columns.json") as f:
        columns = json.load(f)
    return model, columns

model, FEATURE_COLUMNS = load_model()

# Grouped for a friendlier UI (standard UCI Phishing Dataset categories)
GROUPS = {
    "Address Bar Based Features": [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
        "Domain_registeration_length", "Favicon", "port", "HTTPS_token",
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

st.title("🛡️ Phishing Website Detector")
st.caption("Machine Learning-based classifier · Project #23 · Built for Ochuma Chambers")
st.write(
    "This tool uses a trained **Random Forest** model (97.4% accuracy, 0.998 ROC-AUC on held-out test data) "
    "to classify a website as **Legitimate** or **Phishing** based on 30 structural/metadata features."
)

st.divider()

# ---------- Example presets ----------
col1, col2 = st.columns(2)
example_legit = {c: 1 for c in FEATURE_COLUMNS}
example_phish = {c: -1 for c in FEATURE_COLUMNS}

if "feature_values" not in st.session_state:
    st.session_state["feature_values"] = {c: 0 for c in FEATURE_COLUMNS}

with col1:
    if st.button("✅ Load 'Typical Legitimate' Example", use_container_width=True):
        st.session_state["feature_values"] = dict(example_legit)
with col2:
    if st.button("🚩 Load 'Typical Phishing' Example", use_container_width=True):
        st.session_state["feature_values"] = dict(example_phish)

st.divider()
st.subheader("Website Feature Values")
st.caption("For each feature: **1 = Safe indicator, 0 = Suspicious, -1 = Risky indicator** (as encoded in the UCI Phishing Websites Dataset).")

for group_name, features in GROUPS.items():
    with st.expander(group_name, expanded=(group_name == "Address Bar Based Features")):
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                st.session_state["feature_values"][feat] = st.select_slider(
                    feat, options=[-1, 0, 1],
                    value=st.session_state["feature_values"].get(feat, 0),
                    key=f"slider_{feat}",
                )

st.divider()

if st.button("🔍 Classify This Website", type="primary", use_container_width=True):
    input_vector = np.array([[st.session_state["feature_values"][c] for c in FEATURE_COLUMNS]])
    prediction = model.predict(input_vector)[0]
    proba = model.predict_proba(input_vector)[0]
    classes = list(model.classes_)

    is_legit = prediction == 1

    if is_legit:
        confidence = proba[classes.index(1)] * 100
        st.success(f"### ✅ Legitimate Website\n**Confidence: {confidence:.1f}%**")
    else:
        confidence = proba[classes.index(-1)] * 100
        st.error(f"### 🚩 Phishing Website Detected\n**Confidence: {confidence:.1f}%**")

    with st.expander("See raw probability breakdown"):
        for c, p in zip(classes, proba):
            label = "Legitimate" if c == 1 else "Phishing"
            st.write(f"{label}: {p*100:.2f}%")

st.divider()
st.caption(
    "Model: Random Forest (200 trees) · Trained on UCI Phishing Websites Dataset (11,055 records, 30 features) · "
    "Project #23 — Phishing Website Detection Using Machine Learning"
)
