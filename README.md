# Indonesian Food Image Classification
(PyTorch + Classical Machine Learning)

## Problem Description
This project solves a multiclass image classification problem:
given an image of Indonesian food, the goal is to predict the food category.

Potential use cases include:
- food recommendation systems
- restaurant menu digitization
- dietary tracking applications

---

## Dataset
The project uses the Indonesian Food Classification Dataset from Kaggle.
Images are organized in folders by food category.

Due to size and licensing constraints, the dataset is not included in this
repository. Instructions to download the dataset are provided below.

---

## Exploratory Data Analysis
Exploratory data analysis includes:
- Counting images per class
- Visualizing class imbalance
- Inspecting sample images

This analysis highlights dataset imbalance and visual similarity between
certain food categories.

---

## Feature Engineering
Images are converted into numerical features using a pretrained
MobileNetV2 convolutional neural network implemented in PyTorch.

- The CNN is used strictly as a frozen feature extractor
- No fine-tuning is performed
- Each image is represented by a 1280-dimensional embedding

Extracted embeddings are cached and reused to ensure reproducibility.

---

## Models
Two supervised machine learning models are trained on identical embeddings:
1. Logistic Regression (multinomial)
2. Random Forest Classifier

This allows a fair comparison between linear and non-linear models.

---

## Hyperparameter Tuning
Hyperparameters are optimized using GridSearchCV with 3-fold
cross-validation. Search spaces are constrained to ensure CPU-only execution.

---

## Evaluation
Models are evaluated using:
- Accuracy
- Confusion matrix

---

## Backend API
The project includes a FastAPI backend that serves the trained models
via an HTTP API.

The backend:
- Loads the trained models
- Accepts image uploads
- Returns predicted class and probability

### Architecture Diagram
+———––+        HTTP        +——————+
|  Web Client |  <–––––––> |  FastAPI Backend |
| (Streamlit) |  <–––––––> |  Model Inference |
+———––+                   +——————+

The Streamlit application acts as a frontend and communicates with
the backend for inference.

---

## Deployment
The project is deployed as:
- A FastAPI backend (model serving)
- A Streamlit frontend (user interface)

This satisfies the Machine Learning Zoomcamp requirement that models
must be deployed via a backend service.

---

## Reproducibility
- All random seeds are fixed
- Training and evaluation logic is implemented in standalone scripts
- Clear instructions are provided to reproduce results

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Locally

```bash
# Start backend
uvicorn backend.api:app --host 0.0.0.0 --port 8000

# Start frontend
streamlit run app.py
```

## Docker

```bash
docker build -t food-classifier .
docker run -p 8501:8501 -p 8000:8000 food-classifier
```

## Docker compose

```bash
docker compose up -d
```

## Re-Train Machine Learning

```bash
python src/download_dataset.py
python src/extract_embeddings.py
python src/train_logreg.py
python src/train_random_forest.py
```

## Model Evaluation

```bash
python src/evaluate.py
```

## EDA

```bash
python src/eda_images.py
```