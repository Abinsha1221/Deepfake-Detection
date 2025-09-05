from flask import Blueprint, render_template, request, jsonify, g
from app.models import DeepfakeDetector
from app.utils import save_video
import os

main = Blueprint('main', __name__)
# detector = DeepfakeDetector()

def get_detector():
    if 'detector' not in g:
        g.detector = DeepfakeDetector()
    return g.detector

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    video_path = save_video(video_file)
    if not video_path:
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        detector = get_detector()
        result = detector.analyze_video(video_path)
        print("\n\n\n\n Next\n\n\n\n\n\n")
        # Ensure all values are JSON serializable
        response = {
            'is_deepfake': bool(result['is_deepfake']),
            'confidence': float(result['confidence']),
            'message': str(result['message'])
        }
        # Clean up uploaded file
        os.remove(video_path)
        return jsonify(response)
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e)}), 500

