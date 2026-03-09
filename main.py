# Import data preprocessing functions
from src.data_preprocessing import load_data, clean_data, encode_data

# Import EDA visualization functions
from src.eda import (
    churn_distribution,
    correlation_heatmap,
    contract_vs_churn,
    monthly_charges_vs_churn,
    tenure_distribution
)

# Import model training functions
from src.train_model import (
    split_data,
    train_logistic,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    evaluate_model,
    feature_importance,
    compare_models
)


def main():

    print("\nStarting Customer Churn Prediction Project...\n")

    # -----------------------------
    # STEP 1: Load Dataset
    # -----------------------------
    df = load_data("data/telco_churn.csv")

    print("Dataset Loaded Successfully\n")

    print("First 5 Rows of Dataset:\n")
    print(df.head())

    print("\nDataset Shape:", df.shape)


    # -----------------------------
    # STEP 2: Data Cleaning
    # -----------------------------
    df = clean_data(df)

    print("\nData Cleaning Completed")


    # -----------------------------
    # STEP 3: Encode Categorical Data
    # -----------------------------
    df = encode_data(df)

    print("Categorical Encoding Completed")


    # -----------------------------
    # STEP 4: Exploratory Data Analysis
    # -----------------------------
    print("\nGenerating Exploratory Data Analysis Visualizations...\n")

    churn_distribution(df)

    correlation_heatmap(df)

    contract_vs_churn(df)

    monthly_charges_vs_churn(df)

    tenure_distribution(df)


    # -----------------------------
    # STEP 5: Split Dataset
    # -----------------------------
    print("\nSplitting dataset into Train and Test sets...")

    X_train, X_test, y_train, y_test = split_data(df)

    print("Data Split Completed")


    # -----------------------------
    # STEP 6: Train Logistic Regression
    # -----------------------------
    print("\nTraining Logistic Regression Model...\n")

    logistic_model = train_logistic(X_train, y_train)

    evaluate_model(
        logistic_model,
        X_test,
        y_test,
        model_name="logistic_regression"
    )


    # -----------------------------
    # STEP 7: Train Random Forest
    # -----------------------------
    print("\nTraining Random Forest Model...\n")

    rf_model = train_random_forest(X_train, y_train)

    evaluate_model(
        rf_model,
        X_test,
        y_test,
        model_name="random_forest"
    )


    # -----------------------------
    # STEP 8: Train Gradient Boosting
    # -----------------------------
    print("\nTraining Gradient Boosting Model...\n")

    gb_model = train_gradient_boosting(X_train, y_train)

    evaluate_model(
        gb_model,
        X_test,
        y_test,
        model_name="gradient_boosting"
    )


    # -----------------------------
    # STEP 9: Train XGBoost
    # -----------------------------
    print("\nTraining XGBoost Model...\n")

    xgb_model = train_xgboost(X_train, y_train)

    evaluate_model(
        xgb_model,
        X_test,
        y_test,
        model_name="xgboost"
    )


    # -----------------------------
    # STEP 10: Feature Importance
    # -----------------------------
    print("\nGenerating Feature Importance Visualization...\n")

    feature_importance(rf_model, X_train.columns)


    # -----------------------------
    # STEP 11: Model Comparison
    # -----------------------------
    print("\nComparing All Models...\n")

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model
    }

    compare_models(models, X_test, y_test)


    print("\nProject Execution Completed Successfully!\n")


if __name__ == "__main__":
    main()