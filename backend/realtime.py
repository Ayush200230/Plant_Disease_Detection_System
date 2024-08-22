from flask import Blueprint, jsonify
import subprocess

realtime_bp = Blueprint('realtime', __name__)

# Global variable to keep track of the process
process = None

@realtime_bp.route('/start_live_detection', methods=['POST'])
def start_live_detection():
    global process
    if process is not None and process.poll() is None:
        return jsonify({'message': 'Live detection is already running'}), 400

    try:
        process = subprocess.Popen(['python', 'live_detection.py'])
        return jsonify({'message': 'Live detection started'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to start live detection: {e}'}), 500

@realtime_bp.route('/stop_live_detection', methods=['POST'])
def stop_live_detection():
    global process
    if process is None or process.poll() is not None:
        return jsonify({'message': 'Live detection is not running'}), 400

    try:
        process.terminate()
        process.wait()
        process = None
        return jsonify({'message': 'Live detection stopped'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to stop live detection: {e}'}), 500
