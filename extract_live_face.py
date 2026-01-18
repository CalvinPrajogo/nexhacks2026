import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json
import sys

def extract_face_features(image_data):
    """
    Extract facial features from a base64 image or file path.
    Returns a dictionary with the same structure as the database.
    """
    # Initialize MediaPipe Face Landmarker
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    model_path = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Load image from base64 or file path
    if image_data.startswith('data:image'):
        # Base64 from canvas.toDataURL
        base64_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        pil_image = Image.open(io.BytesIO(image_bytes))
    else:
        # File path
        pil_image = Image.open(image_data)
    
    # Convert to RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    image_rgb = np.array(pil_image)
    height, width = image_rgb.shape[:2]
    
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Process the image
    results = detector.detect(mp_image)
    
    if not results.face_landmarks:
        detector.close()
        return {"error": "No face detected in image"}
    
    # Get the first face
    face_landmarks = results.face_landmarks[0]
    
    # Limit to 468 landmarks to match database schema
    if len(face_landmarks) > 468:
        face_landmarks = face_landmarks[:468]
    
    # Calculate bounding box
    x_coords = [lm.x * width for lm in face_landmarks]
    y_coords = [lm.y * height for lm in face_landmarks]
    
    face_x = min(x_coords)
    face_y = min(y_coords)
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)
    
    # Extract key landmarks
    left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
    left_eye_x = np.mean([face_landmarks[i].x * width for i in left_eye_indices])
    left_eye_y = np.mean([face_landmarks[i].y * height for i in left_eye_indices])
    
    right_eye_indices = [362, 263, 387, 386, 385, 384, 398]
    right_eye_x = np.mean([face_landmarks[i].x * width for i in right_eye_indices])
    right_eye_y = np.mean([face_landmarks[i].y * height for i in right_eye_indices])
    
    nose_tip_x = face_landmarks[1].x * width
    nose_tip_y = face_landmarks[1].y * height
    
    mouth_indices = [13, 14, 78, 308]
    mouth_x = np.mean([face_landmarks[i].x * width for i in mouth_indices])
    mouth_y = np.mean([face_landmarks[i].y * height for i in mouth_indices])
    
    left_ear_x = face_landmarks[234].x * width
    left_ear_y = face_landmarks[234].y * height
    right_ear_x = face_landmarks[454].x * width
    right_ear_y = face_landmarks[454].y * height
    
    # Calculate face orientation
    eye_distance = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
    yaw = (face_width / 2 - eye_distance) / face_width * 90
    pitch = ((nose_tip_y - face_y) / face_height - 0.6) * 90
    roll = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi
    face_aspect_ratio = face_width / face_height if face_height > 0 else 0
    
    # Build feature dictionary
    features = {
        'face_location_x': face_x,
        'face_location_y': face_y,
        'face_location_width': face_width,
        'face_location_height': face_height,
        'left_eye_x': left_eye_x,
        'left_eye_y': left_eye_y,
        'right_eye_x': right_eye_x,
        'right_eye_y': right_eye_y,
        'nose_tip_x': nose_tip_x,
        'nose_tip_y': nose_tip_y,
        'mouth_center_x': mouth_x,
        'mouth_center_y': mouth_y,
        'left_ear_x': left_ear_x,
        'left_ear_y': left_ear_y,
        'right_ear_x': right_ear_x,
        'right_ear_y': right_ear_y,
        'face_orientation_yaw': yaw,
        'face_orientation_pitch': pitch,
        'face_orientation_roll': roll,
        'eye_distance': eye_distance,
        'face_aspect_ratio': face_aspect_ratio
    }
    
    # Add all 468 landmarks
    for i, landmark in enumerate(face_landmarks):
        features[f'landmark_{i}_x'] = landmark.x * width
        features[f'landmark_{i}_y'] = landmark.y * height
        features[f'landmark_{i}_z'] = landmark.z
    
    detector.close()
    return features

if __name__ == "__main__":
    # Can be called with image path as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        features = extract_face_features(image_path)
        print(json.dumps(features, indent=2))
    else:
        # Or read base64 from stdin
        image_data = sys.stdin.read().strip()
        features = extract_face_features(image_data)
        print(json.dumps(features, indent=2))
