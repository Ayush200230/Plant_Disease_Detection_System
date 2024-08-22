import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from db import db
from models import User, DetectionHistory

# Path to the dataset
data_dir = 'C:/files/plant-disease-detection/data'
# Path to save the retrained model
model_save_path = 'C:/files/plant-disease-detection/models/plant_disease_model_final.keras'

def load_data():
    images = []
    labels = []
    for label, folder_name in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    image = load_img(file_path, target_size=(128, 128))
                    image = img_to_array(image)
                    image = image / 255.0
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def retrain_model():
    try:
        # Load existing model
        model = load_model(model_save_path)

        # Prepare the dataset
        images, labels = load_data()

        # Split dataset into training and validation sets
        split_index = int(len(images) * 0.8)
        X_train, X_val = images[:split_index], images[split_index:]
        y_train, y_val = labels[:split_index], labels[split_index:]

        # Compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

        # Save the retrained model
        model.save(model_save_path)
        print("Model retrained and saved successfully.")

        # Convert to TensorFlow Lite format
        convert_to_tflite(model_save_path, 'C:/files/plant-disease-detection/models/plant_disease_model_final.tflite')

    except Exception as e:
        print(f"Error during model retraining: {e}")

def check_for_new_data_and_retrain():
    try:
        # Logic to determine if new data is available
        data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if len(data_files) > 0:
            print("New data detected. Retraining the model...")
            retrain_model()
        else:
            print("No new data to retrain.")
    except Exception as e:
        print(f"Error checking for new data: {e}")

def convert_to_tflite(model_path, tflite_model_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

