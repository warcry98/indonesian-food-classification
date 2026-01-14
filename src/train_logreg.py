import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "embeddings")

X_train = np.load(os.path.join(EMBEDDING_DIR, "X_train.npy"))
y_train = np.load(os.path.join(EMBEDDING_DIR, "y_train.npy"))

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs"],
    "max_iter": [1000]
}

model = LogisticRegression(
    random_state=42,
    verbose=1,
)

grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1)
grid.fit(X_train, y_train)

joblib.dump(grid.best_estimator_, os.path.join(SCRIPT_DIRECTORY, "..", "models", "logreg.joblib"))
print("Best Logistic Regression:", grid.best_params_)