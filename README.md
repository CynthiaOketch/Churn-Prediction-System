# Churn Prediction System for Telecom Customers

A machine learning solution that predicts whether a telecom customer will churn based on their demographics, service usage, and account information. The goal is to help telecom companies proactively identify at-risk customers and improve retention strategies.

---

## Project Overview

- **Goal**: Predict customer churn using historical telecom data
- **Dataset**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Tech Stack**:
  - Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
  - Jupyter Notebook for analysis and model building
  - Streamlit for interactive deployment

---

## Project Structure

```
Churn-Prediction-System/
├── app/
│   └── app.py                        # Streamlit app
├── data/
│   ├── Telco-Customer-Churn.csv      # Raw dataset (download from Kaggle — not in repo)
│   └── features.csv                  # Preprocessed features
├── models/
│   ├── churn_pipeline.pkl            # Deployed model (used by the app)
│   └── random_forest_best.pkl        # Best model from GridSearchCV tuning
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_model_optimization.ipynb
├── setup.sh
└── README.md
```

---

## Features

- Exploratory Data Analysis (EDA) with visual insights
- Feature engineering: encoding, scaling, and corrected TotalCharges imputation
- Model training with Random Forest and hyperparameter tuning via GridSearchCV
- Evaluation with precision, recall, F1-score, and ROC AUC
- Streamlit app for real-time churn prediction

### Model performance (Random Forest, optimised — `random_forest_best.pkl`)

Evaluated on a stratified 20% hold-out set (random_state=42). Decision threshold chosen to maximise F1 on the same hold-out set.

| Metric               | Score |
|---------------------|-------|
| ROC AUC              | 0.842 |
| PR AUC               | 0.656 |
| Decision threshold   | 0.481 |
| Accuracy             | 77%   |
| Precision (churn)    | 0.54  |
| Recall (churn)       | 0.75  |
| F1 (churn)           | 0.63  |

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/CynthiaOketch/Churn-Prediction-System.git
   cd Churn-Prediction-System
   ```

2. **Download the dataset**

   Download `Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) and place it in the `data/` directory.

3. **Set up the environment**

   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

4. **Run the app**

   ```bash
   streamlit run app/app.py
   ```
