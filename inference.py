"""
inference.py
-------------
Runs end-to-end inference using trained segmentation and classification models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load models
seg_model = tf.keras.models.load_model("Att_Segnet.keras", compile=False)
clf_model = tf.keras.models.load_model("LesionClassifier.keras", compile=False)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_path, save_mask=True):
    img = preprocess_image(image_path)
    mask = seg_model.predict(img)
    lesion_pred = clf_model.predict(mask)
    class_idx = np.argmax(lesion_pred)
    class_label = "Non-Infected" if class_idx == 0 else "Infected"
    
    if save_mask:
        import cv2
        cv2.imwrite("predicted_mask.png", (mask[0] * 255).astype("uint8"))
    
    return class_label

if __name__ == "__main__":
    test_image = "sample_input.jpg"
    result = predict(test_image)
    print(f"Predicted class: {result}")
