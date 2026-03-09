import shap
from xgboost import XGBClassifier

from data_preprocessing import load_data

X, y = load_data()

model = XGBClassifier()

model.fit(X,y)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)