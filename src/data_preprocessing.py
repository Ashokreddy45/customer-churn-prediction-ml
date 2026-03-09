import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():

    df = pd.read_csv("data/telco_churn.csv")

    # Drop ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns, scaler