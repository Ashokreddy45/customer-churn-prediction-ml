# Customer Churn Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-end Machine Learning project to predict telecom customer churn using multiple ML models with hyperparameter tuning and explainable AI.

---

# Project Overview

Customer churn prediction is an important problem for telecom companies.  
This project analyzes customer behavior and builds machine learning models to predict whether a customer is likely to leave the service.

The project includes:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Multiple machine learning models  
- Hyperparameter tuning  
- Model evaluation  
- Explainable AI using SHAP  
- Deployment using Streamlit and Flask API  

---

# Dataset

**Telco Customer Churn Dataset**

- 7043 customers  
- 21 features  
- Target variable: **Churn**

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the dataset in the following location:

```
data/telco_churn.csv
```

---

# Project Architecture

```
Data Collection
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
(Logistic Regression, Random Forest, XGBoost)
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Model Explainability (SHAP)
      │
      ▼
Deployment (Streamlit + Flask API)
```

---

# Project Structure

```
customer-churn-prediction
│
├── app
│   ├── flask_api.py
│   └── streamlit_app.py
│
├── src
│   ├── data_processing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── hyperparameter_tuning.py
│   └── shap_explainability.py
│
├── outputs
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── confusion_matrix.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Exploratory Data Analysis

### Customer Churn Distribution

![Churn](outputs/churn_distribution.png)

### Feature Correlation Heatmap

![Heatmap](outputs/correlation_heatmap.png)

### Contract Type vs Churn

![Contract](outputs/contract_vs_churn.png)

### Monthly Charges vs Churn

![Charges](outputs/monthly_charges_churn.png)

### Tenure Distribution

![Tenure](outputs/tenure_distribution.png)

---

# Machine Learning Models

The following machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

---

# Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|------|------|------|------|------|
| Logistic Regression | 0.79 | 0.76 | 0.71 | 0.73 |
| Random Forest | 0.84 | 0.82 | 0.78 | 0.80 |
| XGBoost | 0.86 | 0.84 | 0.81 | 0.82 |

### Confusion Matrix

![Confusion](outputs/confusion_matrix.png)

---

# Explainable AI (SHAP)

SHAP values were used to explain how each feature contributes to churn predictions.

Key insights:

- Contract type strongly influences churn
- Monthly charges impact churn probability
- Customers with shorter tenure are more likely to churn

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  
- Flask  

---

# Installation

Clone the repository

```
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Navigate to the project folder

```
cd customer-churn-prediction
```

Install dependencies

```
pip install -r requirements.txt
```

Run the Streamlit application

```
streamlit run app/streamlit_app.py
```

---

# Business Insights

- Customers with **month-to-month contracts** have the highest churn rate.
- Higher **monthly charges** correlate with increased churn probability.
- Customers with **short tenure** are more likely to leave the service.

These insights can help telecom companies develop targeted retention strategies.

---

# Future Improvements

- Deploy the model on a cloud platform
- Add real-time prediction API
- Build a more advanced dashboard
- Integrate automated ML pipelines

---

# Author

**Ashok Reddy Damireddy**

Machine Learning | Data Science | Artificial Intelligence