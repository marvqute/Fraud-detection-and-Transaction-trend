# Credit Card Fraud Detection & Transaction Trends

### Project Overview

This project explores credit card transaction data, performs exploratory data analysis (EDA), applies data balancing techniques using SMOTE, and builds predictive models using Logistic Regression and Random Forest to detect fraudulent transactions. The goal is to identify fraudulent transactions accurately while addressing class imbalance in the dataset.

### Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description:**
  - Contains anonymized credit card transactions.
  - Features: `Time`, `Amount`, and `V1-V28` (PCA transformed features).
  - Imbalanced dataset: Fraudulent transactions (`Class = 1`) are significantly fewer than non-fraudulent ones (`Class = 0`).

---

## Steps in Analysis

### 1. Exploratory Data Analysis (EDA)

- Examined dataset structure and statistics.
- Visualized class distribution.
- Analyzed transaction amount and time distributions.
- Correlation analysis of features.

### 2. Data Preprocessing & Balancing

- Scaled features using `StandardScaler`.
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

### 3. Model Training & Evaluation

- **Logistic Regression**:

  - Used `class_weight='balanced'`, `C=0.5`, and `max_iter=500`.
  - Evaluated with classification report and accuracy.

- **Random Forest Classifier**:

  - Optimized with `n_estimators=50`, `max_depth=10`, and `class_weight='balanced'`.
  - Parallel processing enabled with `n_jobs=-1`.

### 4. Results & Findings

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | \~ 94%   |
| Random Forest       | \~ 98%   |

- Random Forest performed better in fraud detection, but Logistic Regression was computationally efficient.









