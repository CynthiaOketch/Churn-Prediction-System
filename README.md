# Churn Prediction System for Telecom Customers

This project is a machine learning solution designed to predict whether a telecom customer will churn based on their demographics, service usage, and account information. It aims to help telecom companies proactively identify at-risk customers and improve retention strategies.

---

## Project Overview

- **Goal**: Predict customer churn using historical telecom data
- **Dataset**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Tech Stack**:  
  - Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost)  
  - Jupyter Notebook for analysis and model building  
  - Streamlit for interactive dashboard deployment

---

## Project Structure

```bash
Churn-Prediction-System/
├── app/ # Streamlit app files
│ └── app.py
├── data/ # Dataset and derived CSVs
│ ├── Telco-Customer-Churn.csv
│ ├── features.csv
│ └── labels.csv
├── models/ # Saved ML models
│ └── random_forest_best.pkl
├── notebooks/ # Jupyter notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_model_training.ipynb
│ ├── 03_model_testing.ipynb
│ ├── 04_model_evaluation.ipynb
│ └── 05_model_optimization.ipynb
├── venv/ # Virtual environment
└── README.md
```
---

## Features

- Exploratory Data Analysis (EDA) with visual insights
- Feature engineering for better model performance
- Model training using Logistic Regression, Random Forest, and XGBoost
- Evaluation with precision, recall, F1-score, and ROC AUC
- Streamlit app for real-time churn prediction

---

##  How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/CynthiaOketch/churn-prediction.git
   cd churn-prediction
   ```
2. **Create a Virtual Environment**
  bash```
  python3 -m venv venv
  source venv/bin/activate
  ```
3. **Install requirements**
  bash```
  ./setup.sh
  ```
4. **Run the app**
  bash```
  streamlit run app/app.py
  ```