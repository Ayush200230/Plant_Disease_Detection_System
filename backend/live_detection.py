'''
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model_path = '../models/plant_disease_model_final.keras'
model = tf.keras.models.load_model(model_path)

# Threshold for confidence level
CONFIDENCE_THRESHOLD = 0.8

# Dynamically load class labels from the subdirectories in the data folder
data_dir = '../data'
class_labels = {i: folder_name for i, folder_name in enumerate(sorted(os.listdir(data_dir))) if os.path.isdir(os.path.join(data_dir, folder_name))}

def process_frame(frame):
    try:
        # Preprocess the image
        image = cv2.resize(frame, (128, 128))
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Perform prediction
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = np.max(prediction, axis=-1)[0]

        # Determine class label
        predicted_class = class_labels.get(predicted_class_index, 'unknown')

        return predicted_class, confidence

    except Exception as e:
        print(f"Error processing frame: {e}")
        return 'unknown', 0.0

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame
        predicted_class, confidence = process_frame(frame)

        # Annotate the frame
        cv2.putText(frame, f'Class: {predicted_class} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Plant Disease Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the TensorFlow Lite model
tflite_model_path = 'C:/files/plant-disease-detection/models/plant_disease_model_final.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Threshold for confidence level
CONFIDENCE_THRESHOLD = 0.8

# Dynamically load class labels from the subdirectories in the data folder
data_dir = 'C:/files/plant-disease-detection/data'
class_labels = {i: folder_name for i, folder_name in enumerate(sorted(os.listdir(data_dir))) if os.path.isdir(os.path.join(data_dir, folder_name))}

def preprocess_image(img):
    # Resize and normalize image to match training preprocessing
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def process_frame(frame):
    try:
        # Preprocess the image
        image = preprocess_image(frame)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = np.max(prediction, axis=-1)[0]

        # Determine class label
        predicted_class = class_labels.get(predicted_class_index, 'unknown')

        return predicted_class, confidence

    except Exception as e:
        print(f"Error processing frame: {e}")
        return 'unknown', 0.0

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame
        predicted_class, confidence = process_frame(frame)

        # Annotate the frame
        cv2.putText(frame, f'Class: {predicted_class} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Plant Disease Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
