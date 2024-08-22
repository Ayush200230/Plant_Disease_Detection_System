# backend/models.py
from db import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class DetectionHistory(db.Model):
    __tablename__ = 'detection_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    why_it_happened = db.Column(db.String, nullable=True)  # New field
    remedies = db.Column(db.String, nullable=True)          # New field
    next_steps = db.Column(db.String, nullable=True)        # New field

    user = db.relationship('User', backref='detection_histories')

class Blacklist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), unique=True, nullable=False)
