import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join("models", "churn_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")

st.write("Predict the probability that a telecom customer will churn.")

st.divider()

# -----------------------------
# Input Features
# -----------------------------
tenure = st.slider("Customer Tenure (months)", 0, 72, 12)

monthly_charges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    features = np.array([[tenure, monthly_charges]])

    prediction = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{round(prediction*100,2)} %"
    )

    if prediction > 0.5:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Customer Likely to Stay")