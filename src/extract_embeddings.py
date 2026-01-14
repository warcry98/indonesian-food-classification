from genericpath import isdir
import os
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "raw")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_VAL_DIR = os.path.join(DATA_DIR, "valid")

EMBEDDING_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "embeddings")
os.makedirs(EMBEDDING_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()
model.eval()

def load_images_from_dir(base_dir, class_to_idx, type):
    X, y = [], []

    total_images = 0
    for cls in class_to_idx:
        cls_dir = os.path.join(base_dir, cls)
        if os.path.isdir(cls_dir):
            total_images += len(os.listdir(cls_dir))

    with torch.no_grad():
        with tqdm(
            total=total_images,
            desc=f"Loading Images {type}",
            unit="img",
            ncols=100
        ) as pbar:
            for cls, idx in class_to_idx.items():
                cls_dir = os.path.join(base_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue

                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img).unsqueeze(0)
                    emb = model(img).squeeze().numpy()
                    X.append(emb)
                    y.append(idx)

                    pbar.update(1)

    return np.array(X), np.array(y)

class_names = sorted(os.listdir(DATA_TRAIN_DIR))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

print("📥 Loading images...")

X_train, y_train = load_images_from_dir(DATA_TRAIN_DIR, class_to_idx, "TRAIN")
X_val, y_val = load_images_from_dir(DATA_VAL_DIR, class_to_idx, "VALID")
X_test, y_test = load_images_from_dir(DATA_TEST_DIR, class_to_idx, "TEST")

print("💾 Saving embeddings...")

np.save(os.path.join(EMBEDDING_DIR, "X_train.npy"), X_train)
np.save(os.path.join(EMBEDDING_DIR, "X_val.npy"), X_val)
np.save(os.path.join(EMBEDDING_DIR, "X_test.npy"), X_test)
np.save(os.path.join(EMBEDDING_DIR, "y_train.npy"), y_train)
np.save(os.path.join(EMBEDDING_DIR, "y_val.npy"), y_val)
np.save(os.path.join(EMBEDDING_DIR, "y_test.npy"), y_test)
np.save(os.path.join(EMBEDDING_DIR, "class_names.npy"), class_names)

print("✅ Embedding generation complete.")