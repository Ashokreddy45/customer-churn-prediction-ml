import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)


def split_data(df):
    """
    Split dataset into features and target
    """

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_logistic(X_train, y_train):
    """
    Train Logistic Regression Model
    """

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Model
    """

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train, y_train)

    return rf


def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluate model performance
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{model_name} Accuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))


    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"outputs/{model_name}_confusion_matrix.png")

    plt.show()


def feature_importance(model, feature_names):
    """
    Plot feature importance for Random Forest
    """

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        feature_series = pd.Series(
            importance,
            index=feature_names
        )

        top_features = feature_series.nlargest(10)

        plt.figure(figsize=(8,6))

        top_features.plot(kind="barh")

        plt.title("Top 10 Important Features for Churn Prediction")

        plt.xlabel("Importance Score")

        plt.savefig("outputs/feature_importance.png")

        plt.show()