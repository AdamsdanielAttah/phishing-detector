# 🛡️ Phishing Website Detection Using Machine Learning

A Streamlit-based educational cybersecurity project that uses a trained **Random Forest classifier** to classify website-related features as **Legitimate** or **Phishing**.

The project is designed as an integrated **Artificial Intelligence + Software Project Management + Cybersecurity** PSA project. It combines a machine-learning classifier, an interactive software interface, live URL feature extraction, security considerations, testing and project-management documentation.

## Project Objective

The objective is to demonstrate how machine learning can be integrated into a practical cybersecurity software prototype for phishing website detection.

The project covers:

1. Obtaining and preparing a labelled phishing-websites dataset.
2. Maintaining the exact feature structure required by the model.
3. Training and evaluating a Random Forest classifier.
4. Saving and loading the trained model.
5. Building an interactive Streamlit application.
6. Accepting a URL and extracting available website-related signals.
7. Converting extracted signals into the model's 30-feature format.
8. Classifying the resulting feature vector as Legitimate or Phishing.
9. Applying software project management and cybersecurity principles.

## System Overview

The current application provides **two demonstration modes**.

### 🌐 1. Live URL Scanner

The user enters an HTTP/HTTPS URL.

The application follows this flow:

**URL → URL validation → Feature extraction → 30-feature vector → Random Forest → Prediction**

The scanner attempts to extract available:

- URL characteristics
- Domain/DNS information
- WHOIS information where available
- HTML and JavaScript-related signals
- Redirect and webpage resource indicators

The application displays:

- **Legitimate** or **Phishing**
- Random Forest predicted class probability
- The extracted 30-feature vector

### 🧪 2. Manual Feature Demo

The application also provides a manual mode where the user can directly supply the 30 model features.

This mode is useful for:

- Demonstrating how the AI model receives its inputs.
- Explaining the dataset features during the PSA presentation.
- Testing specific feature combinations.
- Showing the relationship between feature values and the model prediction.

## Important Scope and Limitation

The live URL scanner is an **educational best-effort feature-extraction layer**. It is **not** a perfect real-time recreation of every original UCI dataset feature.

Some UCI features depend on historical, reputation or external information that cannot always be obtained reliably from a live URL. Examples include:

- Page Rank
- Web traffic
- Google Index
- Links pointing to a page
- Historical domain information

Where a feature cannot be reliably obtained, the application uses a conservative/neutral value or best-effort lookup.

> **The prediction is an educational machine-learning result and must not be treated as a guarantee that a website is safe or malicious.**

For a production-grade security scanner, the feature-extraction and security architecture would require additional validation, trusted external intelligence sources, monitoring and independent security testing.

## Dataset

The project follows the **UCI Phishing Websites dataset format** specified for the PSA Topic 23.

The model uses these 30 features:

### Address Bar Based Features

- `having_IP_Address`
- `URL_Length`
- `Shortining_Service`
- `having_At_Symbol`
- `double_slash_redirecting`
- `Prefix_Suffix`
- `having_Sub_Domain`
- `SSLfinal_State`
- `Domain_registeration_length`
- `Favicon`
- `port`
- `HTTPS_token`

### Abnormal-Based Features

- `Request_URL`
- `URL_of_Anchor`
- `Links_in_tags`
- `SFH`
- `Submitting_to_email`
- `Abnormal_URL`

### HTML & JavaScript Based Features

- `Redirect`
- `on_mouseover`
- `RightClick`
- `popUpWidnow`
- `Iframe`

### Domain-Based Features

- `age_of_domain`
- `DNSRecord`
- `web_traffic`
- `Page_Rank`
- `Google_Index`
- `Links_pointing_to_page`
- `Statistical_report`

The exact feature order expected by the saved model is stored in `feature_columns.json`.

Maintaining this order is important because the model expects the same feature structure used during training.

## Machine Learning Model

The project uses a **Random Forest Classifier** for binary classification.

The saved model is:

`phishing_rf_model.pkl`

The model classes are:

- `1` → **Legitimate**
- `-1` → **Phishing**

The application also displays the model's class probability. This represents the Random Forest's predicted probability and should **not** be interpreted as absolute certainty or a guarantee of website safety.

### Training Configuration

The included training pipeline uses:

- Random Forest Classifier
- 200 trees (`n_estimators=200`)
- `random_state=42`
- `n_jobs=-1`
- `class_weight="balanced"`
- Stratified train/test split

## Training and Evaluation

The repository includes `train_model.py`, which documents and reproduces the training workflow when the original dataset is available.

Place the dataset in the project folder and run:

```bash
python train_model.py --data phishing.csv
```

By default, the script expects a target column named `Result`. A different target column can be specified:

```bash
python train_model.py --data phishing.csv --target Result
```

The training script:

- Validates the required 30 features.
- Converts feature values to numeric values.
- Converts common target encodings to `-1 / 1`.
- Removes invalid feature/target rows where necessary.
- Creates a stratified train/test split.
- Trains the Random Forest classifier.
- Reports accuracy.
- Reports precision, recall and F1-score through the classification report.
- Reports ROC-AUC when applicable.
- Saves `phishing_rf_model.pkl`.
- Saves the exact feature order to `feature_columns.json`.

### Model Evaluation

A reproducible evaluation was run on the uploaded `PhishingWebsites.csv` dataset using the same configuration documented in `train_model.py`:

- Dataset: 11,055 rows and 30 input features
- Target: `Result`
- Target mapping: `-1 = Phishing`, `1 = Legitimate`
- Train/test split: 80% / 20%, stratified
- Training rows: 8,844
- Testing rows: 2,211
- Random state: 42
- Random Forest trees: 200
- Class weighting: `balanced`

Verified test-set results:

| Metric | Result |
|---|---:|
| Accuracy | **97.42%** |
| Precision — Phishing | **97.82%** |
| Recall — Phishing | **96.33%** |
| F1-score — Phishing | **97.07%** |
| Precision — Legitimate | **97.11%** |
| Recall — Legitimate | **98.29%** |
| F1-score — Legitimate | **97.70%** |
| ROC-AUC (Legitimate as positive class) | **99.78%** |

### Confusion Matrix

| Actual / Predicted | Phishing | Legitimate |
|---|---:|---:|
| Phishing | 944 | 36 |
| Legitimate | 21 | 1,210 |

The model correctly classified **2,154 of 2,211** held-out test samples. There were **36 phishing websites classified as legitimate** (false negatives for the phishing class) and **21 legitimate websites classified as phishing**.

These results are based on the exact uploaded dataset and the reproducible Random Forest configuration in `train_model.py`. They replace the earlier unverified README figures of 97.4% accuracy and 0.998 ROC-AUC.

## Software Implementation

The application is built with:

- Python
- Streamlit
- Scikit-learn
- Joblib
- NumPy
- Pandas
- Requests
- BeautifulSoup
- tldextract
- python-whois

The main application is `app.py`.

### Live URL Application Flow

1. User enters a URL.
2. The application validates and normalizes the URL.
3. URL characteristics are extracted.
4. DNS/WHOIS information is attempted where available.
5. The webpage is requested with a timeout.
6. Selected HTML/JavaScript indicators are extracted.
7. The signals are arranged into the exact 30-feature order.
8. The Random Forest model performs the classification.
9. The result and model probability are displayed.
10. The extracted features are shown for transparency.

## Cybersecurity Considerations

Because the application performs network requests to user-supplied URLs, the live scanner introduces additional security considerations.

### Current considerations

- HTTP/HTTPS URL validation
- Request timeouts
- Exception handling
- Conservative handling of unavailable features
- Clear warning that model output is not a guaranteed security verdict
- Version-controlled source code and model files

### Production Security Improvements

Before using the scanner as a public production service, additional protections should be implemented, especially against **Server-Side Request Forgery (SSRF)**.

Recommended controls include:

- Block localhost and loopback addresses.
- Block private/internal IP ranges.
- Validate DNS resolution before making requests.
- Re-check redirected destinations.
- Restrict outbound network access.
- Apply strict request timeouts and size limits.
- Avoid unnecessarily storing submitted URLs.
- Monitor suspicious request behaviour.
- Review and update third-party dependencies.

## Software Project Management

The project is structured as an integrated PSA project rather than only an AI model.

Project-management activities include:

- Project charter
- Stakeholder identification
- Requirements engineering
- Scope definition
- Work Breakdown Structure (WBS)
- Resource allocation
- Scheduling and milestones
- Risk management
- Quality assurance and quality control
- Change control
- Monitoring and project closure

A major controlled scope extension was the addition of the live URL scanner to the original manual feature demonstration.

## Secure SDLC

Security is considered throughout the software lifecycle:

**Requirements → Design → Development → Testing → Deployment → Monitoring**

Security activities include:

- Security requirements
- Threat modelling
- Input validation
- Safe outbound request handling
- Error handling
- Security testing
- Dependency review
- Monitoring considerations
- Incident response planning

## Threats Considered

| Asset/Component | Threat | Example Control |
|---|---|---|
| Model file | Model tampering | Version control and repository access control |
| Feature configuration | Incorrect/tampered feature order | Validation and code review |
| URL input | Malicious input | URL validation |
| Outbound requests | SSRF | Private/loopback blocking and redirect validation |
| User privacy | URL/data exposure | Minimize unnecessary logging |
| Prediction result | Misinterpretation | Display limitations and model probability |
| Dependencies | Supply-chain vulnerabilities | Dependency review and updates |

## Testing

Testing covers both the software and AI components.

### Functional Testing

Examples include:

- Valid URL input
- Invalid URL input
- Unreachable website
- Redirecting website
- Manual feature input
- Prediction output
- Feature-order validation

### AI/Model Evaluation

The model should be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable

False negatives are particularly important in phishing detection because a phishing website incorrectly classified as legitimate could expose a user to risk.

### Security Testing

Security testing should include:

- Malformed URL testing
- Unsupported URL schemes
- Timeout handling
- Redirect handling
- Error handling
- SSRF testing for public deployments
- Dependency/security review

## Deployment

The application can be run locally or deployed to a compatible Streamlit hosting environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

## Repository Structure

```text
phishing-detector/
├── app.py
├── train_model.py
├── feature_columns.json
├── phishing_rf_model.pkl
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

## Project Limitations

1. The live scanner cannot perfectly reproduce every historical or external UCI feature in real time.
2. Some domain/reputation information may be unavailable or unreliable during a live scan.
3. WHOIS information can be unavailable or slow.
4. Model probabilities are not guarantees of safety.
5. The project is an academic prototype rather than a production security service.
6. Actual model performance should only be reported from a reproducible training/evaluation run.
7. Public deployment requires additional SSRF protections and security hardening.

## Future Improvements

1. Implement a fully SSRF-safe URL fetching architecture.
2. Improve real-time extraction of domain and reputation features.
3. Add trusted external threat-intelligence sources where appropriate.
4. Add confusion matrix and ROC curve visualizations.
5. Add automated unit and integration tests.
6. Add model explainability.
7. Add model drift and performance monitoring.
8. Add periodic retraining using newly labelled data.
9. Improve logging and incident-response capabilities.
10. Conduct independent security testing before real-world use.

## Educational Use

This project is intended for **academic demonstration and learning** in Artificial Intelligence, Software Project Management and Cybersecurity.

It should **not** be treated as a standalone security product or as the sole basis for deciding whether a website is safe.

## Project Files

- `app.py` — Streamlit application and live URL feature extraction.
- `train_model.py` — model training and evaluation pipeline.
- `feature_columns.json` — exact 30-feature model input order.
- `phishing_rf_model.pkl` — trained Random Forest model.
- `requirements.txt` — Python dependencies.
- `runtime.txt` — Python runtime configuration.
- `.gitignore` — files excluded from version control.

## Project Repository

**AdamsdanielAttah/phishing-detector**

This repository contains the software implementation and supporting machine-learning files for the PSA project.
