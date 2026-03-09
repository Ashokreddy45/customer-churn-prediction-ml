import matplotlib.pyplot as plt
import seaborn as sns
import os

# ensure output folder exists
os.makedirs("outputs", exist_ok=True)


def churn_distribution(df):

    plt.figure(figsize=(6,4))
    sns.countplot(x="Churn", data=df)

    plt.title("Customer Churn Distribution")

    plt.savefig("outputs/churn_distribution.png")

    plt.show()


def correlation_heatmap(df):

    plt.figure(figsize=(14,10))

    sns.heatmap(df.corr(), cmap="coolwarm")

    plt.title("Feature Correlation Heatmap")

    plt.savefig("outputs/correlation_heatmap.png")

    plt.show()


# ---------- ADVANCED VISUALIZATION 1 ----------
def contract_vs_churn(df):

    plt.figure(figsize=(7,5))

    sns.countplot(x="Contract", hue="Churn", data=df)

    plt.title("Churn by Contract Type")

    plt.savefig("outputs/contract_vs_churn.png")

    plt.show()


# ---------- ADVANCED VISUALIZATION 2 ----------
def monthly_charges_vs_churn(df):

    plt.figure(figsize=(7,5))

    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)

    plt.title("Monthly Charges vs Churn")

    plt.savefig("outputs/monthly_charges_churn.png")

    plt.show()


# ---------- ADVANCED VISUALIZATION 3 ----------
def tenure_distribution(df):

    plt.figure(figsize=(7,5))

    sns.histplot(df["tenure"], bins=30)

    plt.title("Customer Tenure Distribution")

    plt.savefig("outputs/tenure_distribution.png")

    plt.show()