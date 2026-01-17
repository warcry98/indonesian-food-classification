import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "emneddings")
MODEL_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "models")

X_test = np.load(os.path.join(EMBEDDING_DIR, "X_test.npy"))
y_test = np.load(os.path.join(EMBEDDING_DIR, "y_test.npy"))

class_names = np.load(os.path.join(EMBEDDING_DIR, "class_names.npy"), allow_pickle=True)

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logreg.joblib")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib")),
}

for name, model in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for cls, a in zip(class_names, per_class_acc):
        print(f"{cls}: {a:.2%}")