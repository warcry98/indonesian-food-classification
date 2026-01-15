import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "embeddings")

X_train = np.load(os.path.join(EMBEDDING_DIR, "X_train.npy"))
y_train = np.load(os.path.join(EMBEDDING_DIR, "y_train.npy"))

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    verbose=1
)

grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

joblib.dump(grid.best_estimator_, os.path.join(SCRIPT_DIRECTORY, "..", "models", "random_forest.joblib"))
print("Best Random Forest:", grid.best_params_)