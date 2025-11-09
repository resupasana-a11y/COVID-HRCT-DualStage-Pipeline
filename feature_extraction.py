"""
feature_extraction.py
---------------------
Uses the trained segmentation model to extract lesion-based features
for binary classification (Infected vs Non-Infected). A similar approach can be followed for multiclass classification by adding the corresponding image folders and label files.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained segmentation model
seg_model = tf.keras.models.load_model("Att_Segnet.keras", compile=False)

# Define dataset directories
infected_dataset_dir = "dataset/OG"
non_infected_dataset_dir = "dataset/new_augmented_images"

# Preprocessing
datagen = ImageDataGenerator(rescale=1.0/255.0)

infected_gen = datagen.flow_from_directory(
    infected_dataset_dir,
    target_size=(256, 256),
    class_mode=None,
    batch_size=16,
    color_mode="rgb",
    shuffle=False
)

non_infected_gen = datagen.flow_from_directory(
    non_infected_dataset_dir,
    target_size=(256, 256),
    class_mode=None,
    batch_size=16,
    color_mode="rgb",
    shuffle=False
)

# Predict segmented lesions
X_infected = seg_model.predict(infected_gen)
X_non_infected = seg_model.predict(non_infected_gen)

# Generate binary labels
y_infected = np.zeros(len(X_infected))  # 0 = infected
y_non_infected = np.ones(len(X_non_infected))  # 1 = non-infected

# Combine segmented data
X = np.concatenate([X_infected, X_non_infected], axis=0)
y = np.concatenate([y_infected, y_non_infected], axis=0)

# Save preprocessed arrays for classifier training
np.savez("segmented_features.npz", X=X, y=y)
print("Saved lesion-based features to segmented_features.npz")
