import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("../models/churn_model.pkl","rb"))

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure",0,72)

monthly = st.number_input("Monthly Charges")

if st.button("Predict"):

    features = np.array([[tenure, monthly]])

    prediction = model.predict_proba(features)[0][1]

    st.write("Churn Probability:", round(prediction,2))