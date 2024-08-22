from flask import Flask
from auth import auth_bp
from db import init_db
from model import model_bp
from history import history_bp
from realtime import realtime_bp
from flask_jwt_extended import JWTManager
import threading
import time
from retrain import check_for_new_data_and_retrain
from models import Blacklist

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600

jwt = JWTManager(app)

# This function will be used to check if the token is blacklisted
@jwt.token_in_blocklist_loader
def check_if_token_in_blacklist(jwt_header, jwt_payload):
    jti = jwt_payload['jti']
    return Blacklist.query.filter_by(jti=jti).first() is not None

init_db(app)

app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(model_bp, url_prefix='/model')
app.register_blueprint(history_bp, url_prefix='/history')
app.register_blueprint(realtime_bp, url_prefix='/realtime')

def background_retraining():
    while True:
        print("Checking for new data...")
        try:
            check_for_new_data_and_retrain()
        except Exception as e:
            print(f"Error during retraining: {e}")
        time.sleep(3600)  # Check every hour

if __name__ == '__main__':
    # Start the background retraining thread
    retrain_thread = threading.Thread(target=background_retraining)
    retrain_thread.daemon = True
    retrain_thread.start()

    # Run the Flask application
    app.run(debug=True)
