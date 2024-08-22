# backend/history.py
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from db import db
from models import DetectionHistory, User, Blacklist

history_bp = Blueprint('history', __name__)

@history_bp.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    jwt_token = get_jwt()
    if Blacklist.query.filter_by(jti=jwt_token['jti']).first():  # Use 'jti' instead of 'token'
        return jsonify({'error': 'Token is blacklisted'}), 401

    username = get_jwt_identity().get('username')
    
    if not username:
        return jsonify({'error': 'Invalid token'}), 401

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    user_id = user.id
    history_entries = DetectionHistory.query.filter_by(user_id=user_id).all()
    
    history_list = [{
        'image_path': h.image_path,
        'prediction': h.prediction,
        'why_it_happened': h.why_it_happened,
        'remedies': h.remedies,
        'next_steps': h.next_steps
    } for h in history_entries]

    return jsonify(history_list), 200
