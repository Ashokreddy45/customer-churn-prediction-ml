import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from xgboost import XGBClassifier


# Create outputs folder
os.makedirs("outputs", exist_ok=True)


def split_data(df):
    """
    Split dataset into train and test
    """

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# ------------------------------
# Train Models
# ------------------------------

def train_logistic(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_gradient_boosting(X_train, y_train):

    model = GradientBoostingClassifier()

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model


# ------------------------------
# Evaluate Model
# ------------------------------

def evaluate_model(model, X_test, y_test, model_name="model"):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n==========================")
    print(model_name)
    print("==========================")

    print("Accuracy:", accuracy)

    print("\nClassification Report\n")

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


    # ROC Curve

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, probs)

        auc = roc_auc_score(y_test, probs)

        plt.figure(figsize=(6,4))

        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

        plt.plot([0,1],[0,1],'--')

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title(f"{model_name} ROC Curve")

        plt.legend()

        plt.savefig(f"outputs/{model_name}_roc_curve.png")

        plt.show()


# ------------------------------
# Feature Importance
# ------------------------------

def feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        feature_series = pd.Series(
            importance,
            index=feature_names
        )

        top_features = feature_series.nlargest(10)

        plt.figure(figsize=(8,6))

        top_features.plot(kind="barh")

        plt.title("Top 10 Important Features")

        plt.xlabel("Importance Score")

        plt.savefig("outputs/feature_importance.png")

        plt.show()


# ------------------------------
# Model Comparison
# ------------------------------

def compare_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        results[name] = acc

    print("\nMODEL COMPARISON")

    for name, score in results.items():

        print(f"{name}: {score:.3f}")