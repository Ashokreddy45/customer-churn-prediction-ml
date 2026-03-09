from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data

X, y = load_data()

params = {
    "n_estimators": [100,200],
    "max_depth": [5,10],
    "min_samples_split": [2,5]
}

model = RandomForestClassifier()

grid = GridSearchCV(model, params, cv=5)

grid.fit(X,y)

print("Best parameters:", grid.best_params_)