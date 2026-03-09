import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")

# ------------------------------------------------
# Load Model
# ------------------------------------------------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()

model = pipeline["model"]
feature_names = pipeline["features"]
scaler = pipeline["scaler"]

# ------------------------------------------------
# SHAP Explainer
# ------------------------------------------------

model_name = str(type(model)).lower()

if "xgb" in model_name or "forest" in model_name or "boost" in model_name:
    explainer = shap.TreeExplainer(model)
else:
    background = np.zeros((1, len(feature_names)))
    explainer = shap.LinearExplainer(model, background)

# ------------------------------------------------
# Page Config
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Intelligence System",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Intelligence System")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

page = st.sidebar.selectbox(
    "Navigation",
    ["EDA Dashboard", "Predict Churn"]
)

# ====================================================
# ADVANCEMENT 5 – INTERACTIVE ANALYTICS DASHBOARD
# ====================================================

if page == "EDA Dashboard":

    st.header("📊 Interactive Churn Analytics Dashboard")

    df = pd.read_csv(DATA_PATH)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            df,
            names="Churn",
            title="Customer Churn Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            nbins=40,
            title="Monthly Charges Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            title="Tenure Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df,
            x="Contract",
            color="Churn",
            title="Contract Type vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

# ====================================================
# PREDICTION PAGE
# ====================================================

if page == "Predict Churn":

    st.header("🔮 Customer Churn Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col3:
        partner = st.selectbox("Partner", ["No", "Yes"])

    col4, col5 = st.columns(2)

    with col4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col5:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    st.subheader("Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No"])

    col4, col5, col6 = st.columns(3)

    with col4:
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])

    with col5:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

    with col6:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    st.subheader("Billing")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )

    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input("Monthly Charges", value=70.0)

    with col5:
        total_charges = st.number_input("Total Charges", value=1500.0)

    # ------------------------------------------------
    # Feature Encoding
    # ------------------------------------------------

    input_dict = {feature: 0 for feature in feature_names}

    def set_feature(name):
        if name in input_dict:
            input_dict[name] = 1

    if gender == "Male":
        set_feature("gender_Male")

    if senior == "Yes":
        set_feature("SeniorCitizen")

    if partner == "Yes":
        set_feature("Partner_Yes")

    if dependents == "Yes":
        set_feature("Dependents_Yes")

    if phone_service == "Yes":
        set_feature("PhoneService_Yes")

    if internet_service == "DSL":
        set_feature("InternetService_DSL")

    if internet_service == "Fiber optic":
        set_feature("InternetService_Fiber optic")

    if online_security == "Yes":
        set_feature("OnlineSecurity_Yes")

    if tech_support == "Yes":
        set_feature("TechSupport_Yes")

    if streaming_tv == "Yes":
        set_feature("StreamingTV_Yes")

    if streaming_movies == "Yes":
        set_feature("StreamingMovies_Yes")

    if contract == "One year":
        set_feature("Contract_One year")

    if contract == "Two year":
        set_feature("Contract_Two year")

    if paperless == "Yes":
        set_feature("PaperlessBilling_Yes")

    if payment_method == "Electronic check":
        set_feature("PaymentMethod_Electronic check")

    if payment_method == "Mailed check":
        set_feature("PaymentMethod_Mailed check")

    if payment_method == "Bank transfer":
        set_feature("PaymentMethod_Bank transfer (automatic)")

    if payment_method == "Credit card":
        set_feature("PaymentMethod_Credit card (automatic)")

    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly_charges
    input_dict["TotalCharges"] = total_charges

    st.divider()

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------

    if st.button("Predict Churn"):

        features = np.array([[input_dict[col] for col in feature_names]])
        features = scaler.transform(features)

        probability = model.predict_proba(features)[0][1]

        if probability < 0.30:
            risk = "Low Risk"
        elif probability < 0.60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        st.metric("Churn Probability", f"{round(probability*100,2)}%")
        st.progress(float(probability))

        # ------------------------------------------------
        # Risk Gauge
        # ------------------------------------------------

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0,100]},
                'steps': [
                    {'range': [0,30], 'color': "lightgreen"},
                    {'range': [30,60], 'color': "yellow"},
                    {'range': [60,100], 'color': "salmon"}
                ]
            }
        ))

        st.plotly_chart(fig)

        # ====================================================
        # ADVANCEMENT 4 – RETENTION RECOMMENDATIONS
        # ====================================================

        st.subheader("🎯 Customer Retention Recommendations")

        recommendations = []

        if monthly_charges > 80:
            recommendations.append("Offer discount on monthly charges")

        if contract == "Month-to-month":
            recommendations.append("Promote yearly contract plans")

        if tech_support == "No":
            recommendations.append("Provide free technical support")

        if tenure < 12:
            recommendations.append("Offer loyalty rewards")

        if internet_service == "Fiber optic":
            recommendations.append("Improve fiber service quality")

        if recommendations:
            for r in recommendations:
                st.write("✔", r)
        else:
            st.success("Customer appears stable.")

        # ====================================================
        # SHAP EXPLAINABILITY
        # ====================================================

        st.subheader("🔍 AI Explanation (SHAP)")

        features_df = pd.DataFrame(features, columns=feature_names)

        if "forest" in model_name or "xgb" in model_name or "boost" in model_name:
            shap_values = explainer(features_df)
            shap_values_single = shap_values[0]
        else:
            shap_values = explainer.shap_values(features_df)
            shap_values_single = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values_single.values if hasattr(shap_values_single, "values") else shap_values_single
        })

        shap_df = shap_df.reindex(
            shap_df["Impact"].abs().sort_values(ascending=False).index
        ).head(10)

        fig, ax = plt.subplots()

        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.invert_yaxis()

        st.pyplot(fig)

        # Waterfall plot

        st.subheader("💡 Prediction Explanation")

        fig = plt.figure()

        if hasattr(shap_values_single, "values"):
            shap.plots.waterfall(shap_values_single, max_display=10, show=False)
        else:
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values_single,
                features_df.iloc[0],
                feature_names=feature_names,
                max_display=10,
                show=False
            )

        st.pyplot(fig)