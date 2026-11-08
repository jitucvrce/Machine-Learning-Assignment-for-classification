import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import joblib

st.set_page_config(page_title="ML Classifier Evaluation Hub", layout="wide")

st.title("📊 ML Model Evaluation Dashboard")
st.write("Upload your test dataset to evaluate the performance of our 5 pre-trained classifiers instantly.")

# Sidebar Controls
st.sidebar.header("📁 Data & Model Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine"
]
selected_model_name = st.sidebar.selectbox("Choose Classification Model", model_options)

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    if 'target' not in df.columns:
        st.error("Error: The uploaded CSV must contain a column named exactly 'target'.")
    else:
        # Separate features and target
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        # Format filename to load model
        model_filename = f"model/{selected_model_name.replace(' ', '_')}.joblib"
        
        try:
            # Load pre-trained model statically without retraining
            model = joblib.load(model_filename)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Compute evaluation metrics
            acc = metrics.accuracy_score(y_test, y_pred)
            auc = metrics.roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            prec = metrics.precision_score(y_test, y_pred, zero_division=0)
            rec = metrics.recall_score(y_test, y_pred, zero_division=0)
            f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
            mcc = metrics.matthews_corrcoef(y_test, y_pred)
            
            # Display Dashboard layout layout split into columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader(f"📈 Performance Metrics: {selected_model_name}")
                metrics_data = {
                    "Evaluation Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],
                    "Value": [f"{acc:.4f}", f"{auc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mcc:.4f}"]
                }
                st.table(pd.DataFrame(metrics_data))
                
            with col2:
                st.subheader("🧩 Confusion Matrix Visualization")
                cm = metrics.confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Bad (0)", "Good (1)"], yticklabels=["Bad (0)", "Good (1)"], ax=ax)
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
            st.divider()
            st.subheader("📋 Detailed Classification Text Report")
            report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
            st.json(report_dict)
            
        except FileNotFoundError:
            st.error(f"Could not locate the saved model file at `{model_filename}`. Ensure your model/ directory is committed to your GitHub branch.")
else:
    st.info("💡 Please upload the generated `test_data.csv` file via the sidebar to populate the evaluation dashboard.")
