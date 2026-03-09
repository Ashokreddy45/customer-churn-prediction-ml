import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Perform data cleaning
    """

    # Remove customerID because it is not useful for prediction
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges column to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df


def encode_data(df):
    """
    Convert categorical columns into numeric
    """

    encoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = encoder.fit_transform(df[column])

    return df