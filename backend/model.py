# # backend/model.py

# import json
# import os
# from flask import Blueprint, request, jsonify
# from werkzeug.utils import secure_filename
# from db import db
# from models import DetectionHistory, User, Blacklist
# from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np

# model_bp = Blueprint('model', __name__)
# model = load_model('C:/files/plant-disease-detection/models/plant_disease_model_final.keras')

# # Threshold for confidence level
# CONFIDENCE_THRESHOLD = 0.8

# # Load disease information from JSON
# with open('C:/files/plant-disease-detection/data/diseases_info.json') as f:
#     disease_info = json.load(f)

# # Dynamically load class labels from the subdirectories in the data folder
# data_dir = 'C:/files/plant-disease-detection/data'
# class_labels = {i: folder_name for i, folder_name in enumerate(sorted(os.listdir(data_dir))) if os.path.isdir(os.path.join(data_dir, folder_name))}

# @model_bp.route('/predict', methods=['POST'])
# @jwt_required()
# def predict():
#     jwt_token = get_jwt()
#     if Blacklist.query.filter_by(jti=jwt_token['jti']).first():
#         return jsonify({'error': 'Token is blacklisted'}), 401

#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join('C:/files/plant-disease-detection/temp', filename)
#     file.save(file_path)

#     try:
#         image = load_img(file_path, target_size=(128, 128))
#         image = img_to_array(image)
#         image = image / 255.0
#         image = np.expand_dims(image, axis=0)

#         prediction = model.predict(image)
#         predicted_class_index = np.argmax(prediction, axis=-1)[0]
#         confidence = np.max(prediction, axis=-1)[0]

#         predicted_class = class_labels.get(predicted_class_index, 'unknown')

#         # Retrieve additional information for the predicted class
#         disease_details = disease_info.get(predicted_class, {
#             "name": "Unknown",
#             "why_it_happened": "Information not available.",
#             "remedies": "Information not available.",
#             "next_steps": "Information not available."
#         })

#         if confidence < CONFIDENCE_THRESHOLD or predicted_class == 'unknown':
#             unknown_dir = os.path.join('C:/files/plant-disease-detection/data', 'unknown')
#             if not os.path.exists(unknown_dir):
#                 os.makedirs(unknown_dir)
#             file.save(os.path.join(unknown_dir, filename))
#         else:
#             class_dir = os.path.join('C:/files/plant-disease-detection/data', predicted_class)
#             if not os.path.exists(class_dir):
#                 os.makedirs(class_dir)
#             file.save(os.path.join(class_dir, filename))

#         # Save history entry with all details
#         user_identity = get_jwt_identity()
#         username = user_identity['username']
#         user = User.query.filter_by(username=username).first()

#         if not user:
#             return jsonify({'error': 'User not found'}), 404

#         history_entry = DetectionHistory(
#             user_id=user.id,
#             image_path=file_path,
#             prediction=str(predicted_class),
#             why_it_happened=disease_details['why_it_happened'],
#             remedies=disease_details['remedies'],
#             next_steps=disease_details['next_steps']
#         )
#         db.session.add(history_entry)
#         db.session.commit()

#         confidence = float(confidence)
#         return jsonify({
#             'prediction': disease_details['name'],
#             'confidence': confidence,
#             'why_it_happened': disease_details['why_it_happened'],
#             'remedies': disease_details['remedies'],
#             'next_steps': disease_details['next_steps']
#         }), 200

#     except Exception as e:
#         # Log the exception if needed
#         return jsonify({'error': f'Error processing image: {e}'}), 500


import json
import os
import numpy as np
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from db import db
from models import DetectionHistory, User, Blacklist
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize blueprint
model_bp = Blueprint('model', __name__)

# Define absolute paths
base_dir = 'C:/files/plant-disease-detection'
model_path = os.path.join(base_dir, 'models', 'plant_disease_model_final.tflite')
data_dir = os.path.join(base_dir, 'data')
temp_dir = os.path.join(base_dir, 'temp')

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Threshold for confidence level
CONFIDENCE_THRESHOLD = 0.8

# Load disease information from JSON
disease_info_path = os.path.join(data_dir, 'diseases_info.json')
if not os.path.exists(disease_info_path):
    raise FileNotFoundError(f"Disease info file not found at {disease_info_path}")

with open(disease_info_path) as f:
    disease_info = json.load(f)

# Dynamically load class labels from the subdirectories in the data folder
class_labels = {i: folder_name for i, folder_name in enumerate(sorted(os.listdir(data_dir))) if os.path.isdir(os.path.join(data_dir, folder_name))}

@model_bp.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    jwt_token = get_jwt()
    if Blacklist.query.filter_by(jti=jwt_token['jti']).first():
        return jsonify({'error': 'Token is blacklisted'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)

    try:
        # Load and preprocess image
        image = load_img(file_path, target_size=(128, 128))
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Set tensor for input
        interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
        interpreter.invoke()

        # Get predictions
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = np.max(prediction, axis=-1)[0]

        predicted_class = class_labels.get(predicted_class_index, 'unknown')

        # Retrieve additional information for the predicted class
        disease_details = disease_info.get(predicted_class, {
            "name": "Unknown",
            "why_it_happened": "Information not available.",
            "remedies": "Information not available.",
            "next_steps": "Information not available."
        })

        # Save the image to the appropriate folder
        if confidence < CONFIDENCE_THRESHOLD or predicted_class == 'unknown':
            unknown_dir = os.path.join(data_dir, 'unknown')
            if not os.path.exists(unknown_dir):
                os.makedirs(unknown_dir)
            file.save(os.path.join(unknown_dir, filename))
        else:
            class_dir = os.path.join(data_dir, predicted_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            file.save(os.path.join(class_dir, filename))

        # Save history entry with all details
        user_identity = get_jwt_identity()
        username = user_identity['username']
        user = User.query.filter_by(username=username).first()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        history_entry = DetectionHistory(
            user_id=user.id,
            image_path=file_path,
            prediction=str(predicted_class),
            why_it_happened=disease_details['why_it_happened'],
            remedies=disease_details['remedies'],
            next_steps=disease_details['next_steps']
        )
        db.session.add(history_entry)
        db.session.commit()

        confidence = float(confidence)
        return jsonify({
            'prediction': disease_details['name'],
            'confidence': confidence,
            'why_it_happened': disease_details['why_it_happened'],
            'remedies': disease_details['remedies'],
            'next_steps': disease_details['next_steps']
        }), 200

    except Exception as e:
        # Log the exception if needed
        return jsonify({'error': f'Error processing image: {e}'}), 500
