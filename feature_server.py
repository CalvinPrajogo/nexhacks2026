from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
from extract_live_face import extract_face_features

app = Flask(__name__)
# Enable CORS for React app with explicit configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:3001"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Directory to store extracted features
FEATURES_DIR = 'live_captures'
os.makedirs(FEATURES_DIR, exist_ok=True)

@app.route('/extract-features', methods=['POST'])
def extract_features():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Extract facial features
        features = extract_face_features(image_data)
        
        if 'error' in features:
            return jsonify(features), 400
        
        # Save features to JSON file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'capture_{timestamp}.json'
        filepath = os.path.join(FEATURES_DIR, filename)
        
        # Add metadata
        features['timestamp'] = datetime.now().isoformat()
        features['filename'] = filename
        
        with open(filepath, 'w') as f:
            json.dump(features, f, indent=2)
        
        return jsonify({
            'success': True,
            'features': features,
            'saved_to': filepath,
            'feature_count': len(features)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-captures', methods=['GET'])
def get_captures():
    """Get list of all captured features"""
    try:
        files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.json')]
        captures = []
        
        for file in sorted(files, reverse=True):  # Most recent first
            filepath = os.path.join(FEATURES_DIR, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                captures.append({
                    'filename': file,
                    'timestamp': data.get('timestamp'),
                    'feature_count': len(data)
                })
        
        return jsonify({
            'count': len(captures),
            'captures': captures
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('🚀 Starting Face Feature Extraction Server...')
    print(f'📁 Features will be saved to: {os.path.abspath(FEATURES_DIR)}')
    app.run(debug=True, port=5001)
