import os
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "raw")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_VAL_DIR = os.path.join(DATA_DIR, "valid")

def class_count_from_dir(base_dir, count_per_class: dict = {}):
    for cls in os.listdir(base_dir):
        count = len(os.listdir(os.path.join(base_dir, cls)))
        if count_per_class.get(cls):
            count_per_class[cls] += count
        else:
            count_per_class[cls] = count
    
    return count_per_class

count_per_class = {}
count_per_class = class_count_from_dir(DATA_TEST_DIR, count_per_class)
count_per_class = class_count_from_dir(DATA_TRAIN_DIR, count_per_class)
count_per_class = class_count_from_dir(DATA_VAL_DIR, count_per_class)

rows = []
for cls, count in count_per_class.items():
    rows.append({"class": cls, "count": count})

df = pd.DataFrame(rows)
df.plot(kind="bar", x="class", y="count", title="Images per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
