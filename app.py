import os
import numpy as np
import zipfile
import random
import mne
import torch
import torch.nn as nn
import transformers
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Fix randomization issues
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
transformers.set_seed(42)

st.title("🧠 EEG-Based Schizophrenia Diagnosis")
st.sidebar.header("Upload EEG Dataset")

# 📌 **Step 1: Upload and Extract EEG Dataset**
uploaded_file = st.sidebar.file_uploader("Upload EEG ZIP file", type=["zip"])

if uploaded_file:
    extract_folder = "extracted_eeg"
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    st.sidebar.success("✅ Files Extracted Successfully!")

    # 📌 **Step 2: Load EEG Data**
    def load_eeg(file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data, _ = raw.get_data(return_times=True)
        return data

    # 📌 **Step 3: Feature Extraction**
    def extract_features(data):
        mean_vals = np.mean(data, axis=1)
        var_vals = np.var(data, axis=1)
        psd_vals = np.log(np.abs(np.fft.fft(data, axis=1))**2)
        entropy = -np.sum((psd_vals / np.sum(psd_vals, axis=1, keepdims=True)) * np.log(psd_vals + 1e-10), axis=1)
        return np.concatenate((mean_vals, var_vals, entropy), axis=0)

    # 📌 **Step 4: Load Dataset**
    file_paths = [os.path.join(extract_folder, file) for file in os.listdir(extract_folder) if file.endswith(".edf")]
    X, y = [], []
    for file in file_paths:
        data = load_eeg(file)
        features = extract_features(data)
        X.append(features)
        y.append(0 if "h" in file else 1)  # 'h' = Healthy, else Schizophrenia

    X = np.array(X)
    y = np.array(y)

    # 📌 **Step 5: Data Normalization**
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 📌 **Step 6: Train-Test Split (Fixed Random Seed)**
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 📌 **Step 7: Data Augmentation (Fixed Random Seed)**
    def augment_data(X, y, num_samples=5, seed=42):
        np.random.seed(seed)
        X_aug, y_aug = [], []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.05, X.shape)
            X_aug.append(X + noise)
            y_aug.append(y)
        return np.vstack(X_aug), np.hstack(y_aug)

    X_train, y_train = augment_data(X_train, y_train)

    # 📌 **Step 8: BERT Feature Encoding**
    st.write("🟡 **Training Model... (BERT Feature Extraction in Progress)**")

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    class BERTFeatureExtractor(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
            self.fc = nn.Linear(768, hidden_dim)

        def forward(self, x):
            text_data = [" ".join(map(str, row[:10])) for row in x.tolist()]
            inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
            outputs = self.bert(**inputs).last_hidden_state.mean(dim=1)
            x = self.fc(outputs)
            return x

    bert_model = BERTFeatureExtractor()

    # Convert EEG features using BERT
    X_train_bert = bert_model(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
    X_test_bert = bert_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

    # 📌 **Step 9: Train XGBoost Model**
    st.write("🟡 **Training XGBoost Model...**")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42)  # Fixed random seed
    xgb_clf.fit(X_train_bert, y_train)

    # 📌 **Step 10: Model Evaluation**
    y_pred = xgb_clf.predict(X_test_bert)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ **Model Accuracy:** {accuracy:.4f}")

    # 📌 **Step 11: Confusion Matrix**
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Schizophrenia"])
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)

    # 📌 **Step 12: Dataset Distribution**
    healthy_count = np.sum(y == 0)
    schizophrenia_count = np.sum(y == 1)

    fig, ax = plt.subplots()
    ax.bar(["Healthy", "Schizophrenia"], [healthy_count, schizophrenia_count], color=["blue", "red"])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Count")
    ax.set_title("Dataset Distribution")
    st.pyplot(fig)

    # 📌 **Step 13: Upload EEG for Prediction**
    st.sidebar.header("Upload EEG File for Prediction")
    pred_file = st.sidebar.file_uploader("Upload a single EEG file (.edf)", type=["edf"])

    if pred_file:
        with open("temp.edf", "wb") as f:
            f.write(pred_file.getbuffer())

        # Load and preprocess EEG data
        pred_data = load_eeg("temp.edf")
        pred_features = extract_features(pred_data)
        pred_features = scaler.transform([pred_features])

        # Convert EEG features using BERT
        pred_features_bert = bert_model(torch.tensor(pred_features, dtype=torch.float32)).detach().numpy()

        # Predict using XGBoost
        pred_label = xgb_clf.predict(pred_features_bert)[0]
        diagnosis = "Healthy" if pred_label == 0 else "Schizophrenia"
        st.sidebar.success(f"🧠 **Prediction:** {diagnosis}")

    # 📌 **Step 14: Brain Diagram Visualization**
    def plot_brain_activity():
        fig, ax = plt.subplots(figsize=(5, 5))
        circle = plt.Circle((0, 0), 1, color='lightblue', fill=True)
        ax.add_patch(circle)
        for i in range(10):
            angle = i * (2 * np.pi / 10)
            x, y = np.cos(angle), np.sin(angle)
            ax.plot(x, y, 'ro' if i % 2 == 0 else 'bo')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.title("Brain Activity Representation")
        plt.axis("off")
        return fig

    st.pyplot(plot_brain_activity())

# 📌 **Run the Streamlit App**
# Start with: `streamlit run app.py`
