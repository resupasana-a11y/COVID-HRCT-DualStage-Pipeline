import numpy as np
from tensorflow.keras import layers, models, optimizers

def build_classifier(input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')#2 for Binary classification; 3 for multiclass
    ])
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data = np.load("segmented_features.npz")
    X, y = data["X"], data["y"]

    # Convert labels to categorical if needed
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=2)

    # Split into train/validation/test (if required)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_classifier()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)
    model.save("LesionClassifier.keras")
