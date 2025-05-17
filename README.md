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
churn-prediction/
├── data/
│ └── Telco-Customer-Churn.csv
├── notebooks/
│ └── 01_data_exploration.ipynb
├── app/
│ └── streamlit_app.py
├── models/
│ └── final_model.pkl
├── README.md
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
