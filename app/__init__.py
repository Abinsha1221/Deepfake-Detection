from flask import Flask
from config import Config
import os

def create_app():
    app = Flask(__name__, 
                static_folder='../static',  # Add this line
                template_folder='templates')
    app.config.from_object(Config)
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    from app.routes import main
    app.register_blueprint(main)
    
    return app

