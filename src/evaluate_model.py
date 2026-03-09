import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_preprocessing import load_data

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = XGBClassifier()

model.fit(X_train,y_train)

probs = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, probs)

auc = roc_auc_score(y_test, probs)

plt.plot(fpr,tpr,label="AUC="+str(round(auc,3)))
plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.savefig("outputs/roc_curve.png")