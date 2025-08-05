import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logger import log_prediction, get_logs

# Load model and test data
with open("sonar_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load saved test data and predictions
test_data = pd.read_csv("test_data.csv")
test_labels = pd.read_csv("test_labels.csv").values.ravel()
test_preds = pd.read_csv("test_preds.csv").values.ravel()

# Streamlit App Config
st.set_page_config(page_title="Sonar Rock vs Mine Prediction", layout="wide")
st.title("🎯 SONAR Rock vs Mine Prediction")

# File Upload
uploaded_file = st.file_uploader("📤 Upload a CSV file with 60 features:", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file, header=None)
        st.subheader("📋 Input Data Preview")
        st.dataframe(input_df, use_container_width=True)

        if input_df.shape[1] != 60:
            st.error("❌ Uploaded file must contain exactly 60 features.")
        else:
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)

            for i in range(len(predictions)):
                result = "Rock (R)" if predictions[i] == 0 else "Mine (M)"
                confidence = round(np.max(probabilities[i]), 2)
                st.success(f"🎯 Prediction: **{result}**")
                st.info(f"🧠 Confidence Score: **{confidence}**")

                log_prediction(str(result), float(confidence), input_df.iloc[i].to_json())

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Prediction History Section
with st.expander("📜 View Prediction History"):
    logs = get_logs()
    if logs:
        history_df = pd.DataFrame(logs, columns=["ID", "Timestamp", "Input Data", "Prediction", "Confidence"])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.write("No predictions logged yet.")

# Performance Metrics
with st.expander("📈 Model Performance on Test Set"):
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)

    st.metric("Accuracy", f"{accuracy:.2f}")
    st.metric("Precision", f"{precision:.2f}")
    st.metric("Recall", f"{recall:.2f}")
    st.metric("F1 Score", f"{f1:.2f}")

# Visualization
with st.expander("📊 Visualize Uploaded Data"):
    if uploaded_file and input_df.shape[1] == 60:
        st.subheader("📌 Histogram of First Row Features")
        fig1, ax1 = plt.subplots()
        input_df.iloc[0].plot(kind='bar', ax=ax1)
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        st.pyplot(fig1)

        st.subheader("📌 Heatmap of Uploaded Data")
        fig2, ax2 = plt.subplots()
        sns.heatmap(input_df, cmap="viridis", ax=ax2)
        st.pyplot(fig2)

        st.subheader("📌 PCA Projection (First 2 Components)")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(input_df)
        fig3, ax3 = plt.subplots()
        ax3.scatter(reduced[:, 0], reduced[:, 1], c='blue', s=50)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        st.pyplot(fig3)
