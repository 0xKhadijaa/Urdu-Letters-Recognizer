import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os
from sklearn.metrics import confusion_matrix, classification_report


DATASET_PATH = "urdu_dataset_augmented"
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Dataset preparation

full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    shuffle=True
)

class_names = full_dataset.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Total Classes:", num_classes)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Saved class_names.json for GUI alignment")


dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = int(0.15 * dataset_size)

train_ds = full_dataset.take(train_size)
remaining_ds = full_dataset.skip(train_size)
val_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size)

print(f"\nTrain Batches      : {len(train_ds)}")
print(f"Validation Batches : {len(val_ds)}")
print(f"Test Batches       : {len(test_ds)}")

# Data preporcessing

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Early stopping 

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=4, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

cnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_model.keras", monitor='val_accuracy', save_best_only=True
)

dnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_dnn_model.keras", monitor='val_accuracy', save_best_only=True
)

# CNN

cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,64,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# DNN

dnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,64,1)),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training

print("\nTRAINING CNN\n")
start_time = time.time()
history_cnn = cnn.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, cnn_checkpoint]
)
cnn_training_time = time.time() - start_time


print("\nTRAINING DNN\n")
start_time = time.time()
history_dnn = dnn.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, dnn_checkpoint]
)
dnn_training_time = time.time() - start_time

# Graphs

def plot_history(history, title):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} Accuracy')
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} Loss')
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.show()

plot_history(history_cnn, "CNN")
plot_history(history_dnn, "DNN")


# Evaluation

def get_predictions(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    return np.array(y_true), np.array(y_pred)

print("\nCNN Results\n")
y_true_cnn, y_pred_cnn = get_predictions(cnn, test_ds)
print(classification_report(y_true_cnn, y_pred_cnn, target_names=class_names))
cnn_cm = confusion_matrix(y_true_cnn, y_pred_cnn)
plt.figure(figsize=(12,10))
sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("CNN Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=90); plt.yticks(rotation=0)
plt.show()

print("\nDNN Results\n")
y_true_dnn, y_pred_dnn = get_predictions(dnn, test_ds)
print(classification_report(y_true_dnn, y_pred_dnn, target_names=class_names))
dnn_cm = confusion_matrix(y_true_dnn, y_pred_dnn)
plt.figure(figsize=(12,10))
sns.heatmap(dnn_cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title("DNN Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=90); plt.yticks(rotation=0)
plt.show()


cnn_loss, cnn_accuracy = cnn.evaluate(test_ds, verbose=0)
dnn_loss, dnn_accuracy = dnn.evaluate(test_ds, verbose=0)

print("\n Result \n")
print(f"CNN Accuracy: {cnn_accuracy:.4f} | Loss: {cnn_loss:.4f} | Params: {cnn.count_params()} | Time: {cnn_training_time:.2f}s")
print(f"DNN Accuracy: {dnn_accuracy:.4f} | Loss: {dnn_loss:.4f} | Params: {dnn.count_params()} | Time: {dnn_training_time:.2f}s")

if cnn_accuracy > dnn_accuracy:
    print("CNN performed better.")
else:
    print("DNN performed better.")

cnn.save("final_cnn_urdu_model.keras")
dnn.save("final_dnn_urdu_model.keras")
print("\n Models saved!!")