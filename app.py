import json
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
import streamlit as st
import tldextract
import whois
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_rf_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"

st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_model():
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

missing = [f for group in GROUPS.values() for f in group if f not in FEATURE_COLUMNS]
if missing or len(FEATURE_COLUMNS) != 30:
    st.error("Feature configuration is invalid.")
    st.write("Missing features:", missing)
    st.stop()

# -------------------------------------------------------------------
# URL FEATURE EXTRACTION
# -------------------------------------------------------------------

SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "shorturl.at", "rebrand.ly",
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "click", "work",
    "zip", "review", "country", "stream",
}


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def has_ip_address(hostname):
    if not hostname:
        return -1
    try:
        socket.inet_aton(hostname)
        return 1
    except OSError:
        return -1


def normalize_url(url):
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def get_domain_info(hostname):
    extracted = tldextract.extract(hostname)
    registered_domain = extracted.registered_domain
    return extracted, registered_domain


def get_url_features(url):
    """Extract URL/domain/HTML signals where safely available.

    The original UCI dataset contains 30 engineered features. Some features
    require historical web/domain services that are not reliably available
    during a live demo, so this function uses conservative approximations and
    clearly marks the limitations in the UI.
    """
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    full_url = url.lower()
    extracted, registered_domain = get_domain_info(hostname)

    features = {feature: 0 for feature in FEATURE_COLUMNS}

    # Address-bar features
    features["having_IP_Address"] = has_ip_address(hostname)
    features["URL_Length"] = 1 if len(url) < 54 else (-1 if len(url) > 75 else 0)
    features["Shortining_Service"] = (
        -1 if hostname in SHORTENING_SERVICES or any(hostname.endswith("." + d) for d in SHORTENING_SERVICES) else 1
    )
    features["having_At_Symbol"] = -1 if "@" in url else 1
    features["double_slash_redirecting"] = -1 if "//" in parsed.path else 1
    features["Prefix_Suffix"] = -1 if "-" in registered_domain else 1
    subdomains = [x for x in hostname.split(".") if x]
    features["having_Sub_Domain"] = -1 if len(subdomains) > 3 else (0 if len(subdomains) == 3 else 1)
    features["SSLfinal_State"] = 1 if parsed.scheme == "https" else -1
    features["Domain_registeration_length"] = 0
    features["Favicon"] = 0
    features["port"] = 1 if parsed.port in (None, 80, 443) else -1
    features["HTTPS_token"] = -1 if "https" in hostname and parsed.scheme != "https" else 1

    # Conservative URL-only approximations before page retrieval
    features["Abnormal_URL"] = -1 if hostname and registered_domain and registered_domain not in hostname else 1

    # Domain-based approximations
    try:
        dns_exists = bool(socket.gethostbyname(hostname)) if hostname else False
    except Exception:
        dns_exists = False
    features["DNSRecord"] = 1 if dns_exists else -1

    features["Google_Index"] = 0
    features["Page_Rank"] = 0
    features["web_traffic"] = 0
    features["age_of_domain"] = 0
    features["Links_pointing_to_page"] = 0
    features["Statistical_report"] = 0

    # Try fetching the page for HTML-based signals.
    response = None
    try:
        response = requests.get(
            url,
            timeout=8,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (PhishingDetectorEducationalDemo/1.0)"},
        )
    except requests.RequestException:
        pass

    if response is not None:
        final_url = response.url.lower()
        soup = BeautifulSoup(response.text, "html.parser")

        features["Redirect"] = -1 if len(response.history) > 2 else 1
        features["Request_URL"] = 1
        features["URL_of_Anchor"] = 1
        features["Links_in_tags"] = 1
        features["SFH"] = 1
        features["Submitting_to_email"] = 1
        features["on_mouseover"] = -1 if "onmouseover" in response.text.lower() else 1
        features["RightClick"] = -1 if "contextmenu" in response.text.lower() else 1
        features["popUpWidnow"] = -1 if re.search(r"(window\.open|popup)", response.text, re.I) else 1
        features["Iframe"] = -1 if soup.find("iframe") else 1

        # External resource ratio: a simple approximation of Request_URL.
        resources = soup.find_all(["img", "script", "link"])
        if resources:
            external = 0
            for tag in resources:
                attr = tag.get("src") or tag.get("href") or ""
                if attr.startswith("http") and registered_domain not in attr.lower():
                    external += 1
            ratio = external / len(resources)
            features["Request_URL"] = -1 if ratio > 0.61 else (0 if ratio > 0.22 else 1)

        anchors = soup.find_all("a", href=True)
        if anchors:
            external = sum(
                1 for a in anchors
                if a["href"].startswith("http") and registered_domain not in a["href"].lower()
            )
            ratio = external / len(anchors)
            features["URL_of_Anchor"] = -1 if ratio > 0.67 else (0 if ratio > 0.31 else 1)

        # Form action / email submission heuristic.
        forms = soup.find_all("form", action=True)
        if forms:
            suspicious_forms = sum(
                1 for form in forms
                if "mailto:" in form.get("action", "").lower()
                or (form.get("action", "").startswith("http")
                    and registered_domain not in form.get("action", "").lower())
            )
            features["SFH"] = -1 if suspicious_forms else 1
            features["Submitting_to_email"] = -1 if any(
                "mailto:" in form.get("action", "").lower() for form in forms
            ) else 1

        features["Abnormal_URL"] = -1 if final_url != url.lower() and registered_domain not in final_url else features["Abnormal_URL"]

    # Domain registration age when WHOIS is available.
    try:
        info = whois.whois(registered_domain)
        creation = info.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            if getattr(creation, "tzinfo", None) is None:
                creation = creation.replace(tzinfo=timezone.utc)
            age_days = (now - creation).days
            features["age_of_domain"] = 1 if age_days > 365 else (-1 if age_days < 90 else 0)
        else:
            features["age_of_domain"] = 0
        features["Domain_registeration_length"] = 1
    except Exception:
        features["age_of_domain"] = 0
        features["Domain_registeration_length"] = 0

    return features


def predict_from_features(features):
    vector = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=int)
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    probability = probabilities[classes.index(prediction)] * 100
    return prediction, probability, dict(zip(classes, probabilities))


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

st.title("🛡️ Phishing Website Detector")
st.caption("Project #23 · Random Forest Machine Learning Classifier")

tab_url, tab_manual = st.tabs(["🌐 Scan a URL", "🧪 Manual Feature Demo"])

with tab_url:
    st.subheader("Scan a Website URL")
    st.write(
        "Enter a website address. The app extracts URL, domain and available "
        "HTML signals, converts them to the model's 30-feature format, and "
        "then runs the Random Forest classifier."
    )

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        help="Only scan websites you are authorized to access. This educational scanner performs a normal HTTP request.",
    )

    if st.button("🔍 Scan URL", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                normalized = normalize_url(url)
                parsed = urlparse(normalized)

                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    st.error("Please enter a valid HTTP/HTTPS URL.")
                else:
                    with st.spinner("Extracting website features..."):
                        extracted_features = get_url_features(normalized)
                        prediction, probability, probabilities = predict_from_features(extracted_features)

                    if prediction == 1:
                        st.success(
                            f"### ✅ Model classification: Legitimate\n"
                            f"**Predicted probability: {probability:.1f}%**"
                        )
                    elif prediction == -1:
                        st.error(
                            f"### 🚩 Model classification: Phishing\n"
                            f"**Predicted probability: {probability:.1f}%**"
                        )
                    else:
                        st.warning(f"Unknown model class: {prediction}")

                    st.subheader("Extracted Features")
                    feature_rows = [
                        {"Feature": feature, "Value": extracted_features[feature]}
                        for feature in FEATURE_COLUMNS
                    ]
                    st.dataframe(feature_rows, use_container_width=True, hide_index=True)

                    with st.expander("Probability breakdown"):
                        for class_value, prob in probabilities.items():
                            label = "Legitimate" if class_value == 1 else "Phishing"
                            st.write(f"**{label}:** {prob * 100:.2f}%")

                    st.warning(
                        "Important: live feature extraction uses safe heuristics and "
                        "best-effort WHOIS/HTML checks. Several UCI features depend on "
                        "historical or external signals, so this URL scanner is an "
                        "educational approximation and should not be treated as a production "
                        "security verdict."
                    )
            except Exception as exc:
                st.error("The URL scan failed.")
                st.code(str(exc))

with tab_manual:
    st.info(
        "This mode demonstrates the original ML workflow by letting you supply "
        "the 30 encoded features directly."
    )

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

    st.caption(
        "**1 = legitimate indicator, 0 = neutral/unknown, -1 = phishing indicator.**"
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

    if st.button("🔍 Classify Features", type="primary", use_container_width=True):
        try:
            prediction, probability, probabilities = predict_from_features(
                st.session_state.feature_values
            )
            if prediction == 1:
                st.success(f"### ✅ Classified as Legitimate\n**Probability: {probability:.1f}%**")
            else:
                st.error(f"### 🚩 Classified as Phishing\n**Probability: {probability:.1f}%**")
            with st.expander("Probability breakdown"):
                for class_value, prob in probabilities.items():
                    label = "Legitimate" if class_value == 1 else "Phishing"
                    st.write(f"**{label}:** {prob * 100:.2f}%")
        except Exception as exc:
            st.error("Prediction failed.")
            st.code(str(exc))

st.divider()
st.caption(
    "Random Forest · 30 UCI phishing features · Educational demonstration. "
    "Do not use the result as the sole basis for deciding whether a website is safe."
)
