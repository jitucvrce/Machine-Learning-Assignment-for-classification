import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import joblib

# 1. Download and load the pre-screened dataset
print("Fetching Wine Quality Dataset...")
url = "https://uci.edu"
df = pd.read_csv(url, sep=';')

# 2. Convert to a binary classification problem (Quality >= 7 is 'Good' (1), else 'Bad' (0))
df['target'] = (df['quality'] >= 7).astype(int)
df = df.drop(columns=['quality'])

# Verify constraints: Features >= 12 (11 features + 1 target), Instances >= 500
print(f"Dataset Shape: {df.shape} (Rows: {df.shape[0]}, Features: {df.shape[1]})")

# 3. Split Features and Target
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Standard Scaling (Crucial for KNN, Logistic Regression, and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled sets back to DataFrames to preserve feature names for Streamlit tracking
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Save test data to CSV (Features + Target combined)
test_data = X_test_scaled_df.copy()
test_data['target'] = y_test.values
test_data.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully.")

# 5. Initialize directories
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")

# 6. Initialize the 5 required Classifiers
classifiers = {
    "Logistic_Regression": LogisticRegression(random_state=42),
    "Decision_Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive_Bayes": GaussianNB(),
    "Support_Vector_Machine": SVC(probability=True, random_state=42)
}

# 7. Train, Evaluate and Export Models
performance_records = []

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_scaled_df, y_train)
    
    # Save the trained model file
    joblib.dump(clf, f"model/{name}.joblib")
    
    # Predict on test data
    y_pred = clf.predict(X_test_scaled_df)
    y_proba = clf.predict_proba(X_test_scaled_df)[:, 1]
    
    # Compute the 6 required evaluation metrics
    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_proba)
    prec = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    
    performance_records.append({
        "ML Model Name": name.replace("_", " "),
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

# Output metric matrix printout to copy directly into your README markdown report
summary_df = pd.DataFrame(performance_records)
print("\n--- FINAL EVALUATION METRICS MATRIX ---")
print(summary_df.to_markdown(index=False))
