import sqlite3
import numpy as np
import json
import sys
from extract_live_face import extract_face_features

def calculate_distance(features1, features2):
    """
    Calculate Euclidean distance between two sets of facial landmarks.
    Lower distance = more similar faces.
    """
    # Extract landmark coordinates
    landmarks1 = []
    landmarks2 = []
    
    for i in range(468):
        landmarks1.extend([
            features1.get(f'landmark_{i}_x', 0),
            features1.get(f'landmark_{i}_y', 0),
            features1.get(f'landmark_{i}_z', 0)
        ])
        landmarks2.extend([
            features2.get(f'landmark_{i}_x', 0),
            features2.get(f'landmark_{i}_y', 0),
            features2.get(f'landmark_{i}_z', 0)
        ])
    
    # Normalize by face size to make comparison scale-invariant
    face_size1 = features1.get('face_location_width', 1) * features1.get('face_location_height', 1)
    face_size2 = features2.get('face_location_width', 1) * features2.get('face_location_height', 1)
    
    # Calculate Euclidean distance
    arr1 = np.array(landmarks1) / np.sqrt(face_size1)
    arr2 = np.array(landmarks2) / np.sqrt(face_size2)
    
    distance = np.linalg.norm(arr1 - arr2)
    return distance

def find_match(live_image_data, threshold=100):
    """
    Find matching face in database.
    Returns the best match with distance score.
    """
    # Extract features from live image
    live_features = extract_face_features(live_image_data)
    
    if 'error' in live_features:
        return live_features
    
    # Connect to database
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    
    # Get all faces from database
    cursor.execute("SELECT id, name FROM faces")
    rows = cursor.fetchall()
    
    best_match = None
    best_distance = float('inf')
    
    for face_id, name in rows:
        # Get all features for this face
        db_features = {}
        
        # Get basic features
        cursor.execute("""
            SELECT face_location_x, face_location_y, face_location_width, face_location_height,
                   left_eye_x, left_eye_y, right_eye_x, right_eye_y,
                   nose_tip_x, nose_tip_y, mouth_center_x, mouth_center_y,
                   left_ear_x, left_ear_y, right_ear_x, right_ear_y,
                   face_orientation_yaw, face_orientation_pitch, face_orientation_roll,
                   eye_distance, face_aspect_ratio
            FROM faces WHERE id = ?
        """, (face_id,))
        
        row = cursor.fetchone()
        db_features = {
            'face_location_x': row[0],
            'face_location_y': row[1],
            'face_location_width': row[2],
            'face_location_height': row[3],
            'left_eye_x': row[4],
            'left_eye_y': row[5],
            'right_eye_x': row[6],
            'right_eye_y': row[7],
            'nose_tip_x': row[8],
            'nose_tip_y': row[9],
            'mouth_center_x': row[10],
            'mouth_center_y': row[11],
            'left_ear_x': row[12],
            'left_ear_y': row[13],
            'right_ear_x': row[14],
            'right_ear_y': row[15],
            'face_orientation_yaw': row[16],
            'face_orientation_pitch': row[17],
            'face_orientation_roll': row[18],
            'eye_distance': row[19],
            'face_aspect_ratio': row[20]
        }
        
        # Get landmarks
        for i in range(468):
            cursor.execute(f"""
                SELECT landmark_{i}_x, landmark_{i}_y, landmark_{i}_z
                FROM faces WHERE id = ?
            """, (face_id,))
            lm = cursor.fetchone()
            db_features[f'landmark_{i}_x'] = lm[0]
            db_features[f'landmark_{i}_y'] = lm[1]
            db_features[f'landmark_{i}_z'] = lm[2]
        
        # Calculate distance
        distance = calculate_distance(live_features, db_features)
        
        if distance < best_distance:
            best_distance = distance
            best_match = {
                'id': face_id,
                'name': name,
                'distance': distance,
                'match': distance < threshold
            }
    
    conn.close()
    
    result = {
        'live_features': live_features,
        'best_match': best_match,
        'threshold': threshold
    }
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100
        result = find_match(image_path, threshold)
        print(json.dumps(result, indent=2))
    else:
        # Read base64 from stdin
        image_data = sys.stdin.read().strip()
        result = find_match(image_data)
        print(json.dumps(result, indent=2))
