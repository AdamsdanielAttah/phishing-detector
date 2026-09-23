# 🛡️ Phishing Website Detection Using Machine Learning

A Streamlit-based educational project that uses a trained Random Forest classifier to classify websites as **Legitimate** or **Phishing** from 30 pre-extracted phishing-detection features.

## Project objective

The objective is to demonstrate a complete machine-learning classification workflow:

1. Obtain a labelled phishing-websites dataset.
2. Prepare the data and keep the feature order consistent.
3. Train a Random Forest classifier.
4. Evaluate the model on held-out test data.
5. Save the trained model.
6. Build a Streamlit interface for prediction.

## Important scope

This application is a **feature-based classifier**. The current UI does not automatically open a URL and extract all 30 features.

The prediction flow is:

**Website data → 30 extracted features → Random Forest → Legitimate / Phishing**

For a production-grade URL scanner, an additional feature-extraction layer would be required before the model.

## Dataset

The project is based on the UCI Phishing Websites dataset format, using these 30 features:

- Address-bar features: `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`, `having_Sub_Domain`, `SSLfinal_State`, `Domain_registeration_length`, `Favicon`, `port`, `HTTPS_token`
- Abnormal features: `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email`, `Abnormal_URL`
- HTML/JavaScript features: `Redirect`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe`
- Domain features: `age_of_domain`, `DNSRecord`, `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`, `Statistical_report`

The exact feature order used by the saved model is stored in `feature_columns.json`.

## Model

The repository contains a pre-trained Random Forest model in:

`phishing_rf_model.pkl`

The model predicts the two dataset classes:

- `1` → Legitimate
- `-1` → Phishing

The Streamlit app also displays the model's class probability. This should be interpreted as the Random Forest's predicted probability, not as a guarantee that a website is safe.

## Training and evaluation

The repository now includes `train_model.py` so the training process is documented and reproducible when the original dataset is available.

Place the dataset in the project folder and run:

```bash
python train_model.py --data phishing.csv
```

By default the script expects a target column named `Result`. You can specify another target column with:

```bash
python train_model.py --data phishing.csv --target Result
```

The script:

- validates the required 30 features;
- converts feature values to numeric values;
- converts the target to the expected `-1 / 1` labels;
- creates a stratified train/test split;
- trains a Random Forest;
- reports accuracy, classification report and ROC-AUC when available;
- saves `phishing_rf_model.pkl`;
- saves the exact feature order to `feature_columns.json`.

The original project description reported **97.4% accuracy and 0.998 ROC-AUC on a held-out test set**. Those figures should only be presented as verified results after running the included training/evaluation script on the exact dataset and split used for the project.

## Run the Streamlit application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Repository structure

```text
phishing-detector/
├── app.py
├── train_model.py
├── feature_columns.json
├── phishing_rf_model.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

## Limitations

- The current app requires the 30 features to be supplied; it does not automatically extract them from a URL.
- The saved model alone does not prove how the model was trained, which is why `train_model.py` documents the pipeline.
- Synthetic preset buttons are demonstrations of model input patterns, not real websites.
- A real security product should validate URLs, extract features safely, monitor model performance, handle drift, and be independently tested.

## Future improvements

1. Build an automated URL feature-extraction module.
2. Add a URL input field that feeds extracted features into the model.
3. Add model evaluation plots such as confusion matrix and ROC curve.
4. Add input validation and logging.
5. Add automated tests for feature order and prediction output.

## Educational use

This project is intended for academic demonstration of machine-learning classification and should not be treated as a standalone security product.
