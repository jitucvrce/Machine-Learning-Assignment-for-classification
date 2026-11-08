# White Wine Quality Classification App

## a. Problem Statement
The objective of this project is to model and predict whether a specific batch of white wine qualifies as high quality ("Good") or normal/low quality ("Bad") based on its objective chemical properties. This helps winemakers automate quality control assessments before human sensory panels run final tests. This task is framed as a supervised binary classification problem.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository (Wine Quality Dataset - White Variant)
- **Instances:** 4,898 instances (Exceeds the 500-instance requirement)
- **Features:** 12 total dimensions (11 inputs + 1 target classification metric)
- **Input Variables:**
  1. Fixed acidity
  2. Volatile acidity
  3. Citric acid
  4. Residual sugar
  5. Chlorides
  6. Free sulfur dioxide
  7. Total sulfur dioxide
  8. Density
  9. pH
  10. Sulphates
  11. Alcohol
- **Target Variable:** `target` (Binary flag: `1` if quality score >= 7 representing "Good", `0` if quality score < 7 representing "Bad").

## c. GitHub Repository & Live Deployment Links
- **GitHub Repository Link:** [INSERT_YOUR_GITHUB_REPO_URL_HERE]
- **Live Streamlit App Link:** [INSERT_YOUR_STREAMLIT_COMMUNITY_CLOUD_URL_HERE]

## d. Models Used & Evaluation Metric Matrices

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8031 | 0.7932 | 0.5897 | 0.2222 | 0.3228 | 0.2796 |
| **Decision Tree** | 0.8071 | 0.7686 | 0.5750 | 0.3333 | 0.4220 | 0.3391 |
| **KNN** | 0.8143 | 0.8051 | 0.6042 | 0.4203 | 0.4957 | 0.4074 |
| **Naive Bayes** | 0.7184 | 0.7621 | 0.3951 | 0.6957 | 0.5039 | 0.3664 |
| **Support Vector Machine** | 0.8224 | 0.8276 | 0.6860 | 0.2850 | 0.4027 | 0.3708 |

### Performance Observations
- **Logistic Regression:** Serves as a strong, predictable linear baseline. It yields high accuracy (80.3%) but displays a weak recall score (22.2%), indicating it frequently misses marginal high-quality instances.
- **Decision Tree:** Effectively captures non-linear splits in variable bounds such as `alcohol` content thresholds. However, it displays structural fragility regarding slight boundary adjustments.
- **KNN:** Strong multi-dimensional neighborhood performer. Achieves highly balanced metric weights across F1 and MCC calculations due to uniform distance bounds.
- **Naive Bayes:** Shows the highest single recall output (69.57%) but incurs a severe penalty regarding precision due to its strong assumption of independent features.
- **Support Vector Machine:** Achieves the highest baseline **Accuracy (82.24%) and AUC (0.8276)** across all evaluated frameworks. It maintains clean boundaries with optimized margins.

### 🏆 Overall Winner Selection
**Support Vector Machine (SVM)** is selected as the winner for this specific dataset. It achieves the peak validation Accuracy and Area Under the ROC Curve (AUC), maintaining the highest functional precision boundary without overfitting.
# Machine-Learning-Assignment-for-classification
Machine Learning Assignment for classification
