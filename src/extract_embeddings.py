import os
import numpy as np
import tensorflow as tf
import kagglehub
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/raw"