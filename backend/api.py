from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from PIL import Image
import joblib
import io
import os

app = FastAPI(title="Indonesian Food Classifier API")

cnn = models.mobilenet_v2(weights="IMAGENET1K_V1")
cnn.classifier = torch.nn.Identity()
cnn.eval()

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

models_ml = {
    "logreg": joblib.load(os.path.join(SCRIPT_DIRECTORY, "..", "models", "logreg.joblib")),
    "rf": joblib.load(os.path.join(SCRIPT_DIRECTORY, "..", "models", "random_forest.joblib"))
}

class_names = np.load(os.path.join(SCRIPT_DIRECTORY, "..", "data", "embeddings", "class_names.npy"), allow_pickle=True)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "logreg"
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with torch.no_grad():
        emb = cnn(transform(image).unsqueeze(0)).numpy()
    
    model = models_ml[model_name]
    probs = model.predict_proba(emb)[0]
    pred = int(np.argmax(probs))

    return {
        "predicted_class": class_names[pred],
        "probability": float(probs[pred]),
        "model": model_name
    }