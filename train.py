# ==============================
# Potato Leaf Disease Detection
# ==============================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random
from PIL import Image

# ==============================
# DATASET PATH (CHANGE THIS)
# ==============================
dataset_path = "C:\\Users\\Sripad Jujar\\OneDrive\\Desktop\\intern-project1\\PotatoPlants"  
# Example: C:/Users/YourName/dataset/PotatoPlants

# ==============================
# CHECK CLASSES
# ==============================
classes = os.listdir(dataset_path)
print("Classes:", classes)

# ==============================
# DISPLAY SAMPLE IMAGES
# ==============================
plt.figure(figsize=(10, 5))

for i, cls in enumerate(classes):
    img_path = os.path.join(dataset_path, cls, random.choice(os.listdir(os.path.join(dataset_path, cls))))
    img = Image.open(img_path)

    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.show()

# ==============================
# DATA PREPROCESSING
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==============================
# MODEL BUILDING
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.summary()

# ==============================
# COMPILE MODEL
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# TRAIN MODEL
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ==============================
# PLOT ACCURACY
# ==============================
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()

# ==============================
# EVALUATION
# ==============================
val_loss, val_acc = model.evaluate(val_data)
print("Validation Accuracy:", val_acc)

# ==============================
# PREDICTIONS
# ==============================
Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\\nClassification Report:\\n")
print(classification_report(val_data.classes, y_pred, target_names=classes))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(val_data.classes, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# SAVE MODEL (IMPORTANT)
# ==============================
model.save("model.h5")
print("Model saved as model.h5")