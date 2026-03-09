import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from xgboost import XGBClassifier

from data_preprocessing import load_data


# ------------------------------------------------
# Create folders
# ------------------------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ------------------------------------------------
# Train Models
# ------------------------------------------------

def train_models(X_train, y_train):

    models = {

        "LogisticRegression": LogisticRegression(max_iter=1000),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "GradientBoosting": GradientBoostingClassifier(),

        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


# ------------------------------------------------
# Evaluate Model
# ------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print("\n==============================")
    print(model_name)
    print("==============================")

    print("Accuracy:", accuracy)

    print("\nClassification Report\n")

    print(classification_report(y_test, preds))


    # ----------------------------
    # Confusion Matrix
    # ----------------------------

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6,4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"outputs/{model_name}_confusion_matrix.png")

    plt.close()


    # ----------------------------
    # ROC Curve
    # ----------------------------

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

        plt.close()

    return accuracy


# ------------------------------------------------
# Main Pipeline
# ------------------------------------------------

def main():

    print("\nLoading dataset...")

    X, y, feature_names, scaler = load_data()

    print("Dataset shape:", X.shape)


    # ----------------------------
    # Train Test Split
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    print("\nTraining models...")

    models = train_models(X_train, y_train)


    print("\nEvaluating models...")

    results = {}

    for name, model in models.items():

        acc = evaluate_model(model, X_test, y_test, name)

        results[name] = acc


    # ----------------------------
    # Model Comparison
    # ----------------------------

    print("\nModel Comparison")

    for name, score in results.items():

        print(f"{name}: {score:.3f}")


    # ----------------------------
    # Select Best Model
    # ----------------------------

    best_model_name = "XGBoost"
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")


    # ----------------------------
    # Save Model Pipeline
    # ----------------------------

    pipeline = {
        "model": best_model,
        "features": feature_names,
        "scaler": scaler
    }

    model_path = "models/churn_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to {model_path}")


# ------------------------------------------------

if __name__ == "__main__":
    main()