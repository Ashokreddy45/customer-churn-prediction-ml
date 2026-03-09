# Customer Churn Prediction using Machine Learning

This project predicts telecom customer churn using machine learning techniques.  
The model analyzes customer behavior and identifies patterns that lead to customer attrition.

---

## Dataset

Telco Customer Churn Dataset

• 7043 customers  
• 21 features  
• Target variable: Churn

---

## Project Workflow

1. Data Cleaning
2. Feature Encoding
3. Exploratory Data Analysis
4. Model Training
5. Model Evaluation

---

## Exploratory Data Analysis

### Customer Churn Distribution
![Churn](outputs/churn_distribution.png)

### Feature Correlation
![Heatmap](outputs/correlation_heatmap.png)

### Contract Type vs Churn
![Contract](outputs/contract_vs_churn.png)

### Monthly Charges vs Churn
![Charges](outputs/monthly_charges_churn.png)

### Tenure Distribution
![Tenure](outputs/tenure_distribution.png)

---

## Machine Learning Models

Logistic Regression  
Random Forest Classifier

---

## Model Performance

Random Forest Accuracy: ~87%

### Confusion Matrix

![Confusion](outputs/confusion_matrix.png)

---

## Technologies Used

Python  
Pandas  
NumPy  
Matplotlib  
Seaborn  
Scikit-learn

---

## Business Insights

Customers with month-to-month contracts have higher churn rates.

Higher monthly charges correlate with increased churn probability.

Customers with shorter tenure are more likely to leave the service.

---

## Future Improvements

Deploy the model using Flask API  
Build a customer churn dashboard