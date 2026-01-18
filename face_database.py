import mediapipe as mp
import cv2
import sqlite3
import os
import numpy as np
from PIL import Image

def create_database():
    """Create SQLite database with face features table"""
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    468 face landmark features (x, y, z for each landmark)
    # Plus additional computed features
    landmark_columns = []
    for i in range(468):
        landmark_columns.extend([
            f'landmark_{i}_x REAL',
            f'landmark_{i}_y REAL',
            f'landmark_{i}_z REAL'
        ])
    
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            face_location_x REAL,
            face_location_y REAL,
            face_location_width REAL,
            face_location_height REAL,
            {', '.join(landmark_columns)},
            left_eye_x REAL, left_eye_y REAL,
            right_eye_x REAL, right_eye_y REAL,
            nose_tip_x REAL, nose_tip_y REAL,
            mouth_center_x REAL, mouth_center_y REAL,
            left_ear_x REAL, left_ear_y REAL,
            right_ear_x REAL, right_ear_y REAL,
            face_orientation_yaw REAL,
            face_orientation_pitch REAL,
            face_orientation_roll REAL,
            eye_distance REAL,
            face_aspect_ratio
            bottom_lip_x REAL, bottom_lip_y REAL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database created successfully!")

def extract_and_store_faces(image_dir='face_data/images'):
    """Extract facial features from images and store in database"""
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Get all PNG files from the images directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
    
    if not image_files:
        print(f"No PNG files found in {image_dir}")
        print(f"Please add your face images to: {os.path.abspath(image_dir)}")
        conn.close()
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing: {image_file}")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ⚠️  Could not load {image_file}")
            continue
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Process the image
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print(f"  ⚠️  No face detected in {image_file}")
            continue
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract name from filename (remove .png extension)
        name = os.path.splitext(image_file)[0]
        
        # Calculate bounding box
        x_coords = [lm.x * width for lm in face_landmarks.landmark]
        y_coords = [lm.y * height for lm in face_landmarks.landmark]
        
        face_x = min(x_coords)
        face_y = min(y_coords)
        face_width = max(x_coords) - min(x_coords)
        face_height = max(y_coords) - min(y_coords)
        
        # Extract key landmarks (using MediaPipe face mesh indices)
        # Left eye center (average of left eye landmarks)
        left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
        left_eye_x = np.mean([face_landmarks.landmark[i].x * width for i in left_eye_indices])
        left_eye_y = np.mean([face_landmarks.landmark[i].y * height for i in left_eye_indices])
        
        # Right eye center
        right_eye_indices = [362, 263, 387, 386, 385, 384, 398]
        right_eye_x = np.mean([face_landmarks.landmark[i].x * width for i in right_eye_indices])
        right_eye_y = np.mean([face_landmarks.landmark[i].y * height for i in right_eye_indices])
        
        # Nose tip
        nose_tip_x = face_landmarks.landmark[1].x * width
        nose_tip_y = face_landmarks.landmark[1].y * height
        
        # Mouth center
        mouth_indices = [13, 14, 78, 308]
        mouth_x = np.mean([face_landmarks.landmark[i].x * width for i in mouth_indices])
        mouth_y = np.mean([face_landmarks.landmark[i].y * height for i in mouth_indices])
        
        # Ears (approximate)
        left_ear_x = face_landmarks.landmark[234].x * width
        left_ear_y = face_landmarks.landmark[234].y * height
        right_ear_x = face_landmarks.landmark[454].x * width
        right_ear_y = face_landmarks.landmark[454].y * height
        
        # Calculate face orientation (simplified)
        eye_distance = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
        
        # Yaw (left-right rotation) - based on eye distance vs face width
        yaw = (face_width / 2 - eye_distance) / face_width * 90
        
        # Pitch (up-down tilt) - based on nose position
        pitch = ((nose_tip_y - face_y) / face_height - 0.6) * 90
        
        # Roll (head tilt) - based on eye alignment
        roll = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi
        
        # Face aspect ratio
        face_aspect_ratio = face_width / face_height if face_height > 0 else 0
        
        # Prepare data for insertion
        columns = ['name', 'image_path', 'face_location_x', 'face_location_y', 
                  'face_location_width', 'face_location_height']
        values = [name, image_path, face_x, face_y, face_width, face_height]
        
        # Add all 468 landmarks (x, y, z for each)
        for i, landmark in enumerate(face_landmarks.landmark):
            columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            values.extend([landmark.x * width, landmark.y * height, landmark.z])
        
        # Add computed features
        columns.extend(['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
                       'nose_tip_x', 'nose_tip_y', 'mouth_center_x', 'mouth_center_y',
                       'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y',
                       'face_orientation_yaw', 'face_orientation_pitch', 'face_orientation_roll',
                       'eye_distance', 'face_aspect_ratio'])
        values.extend([left_eye_x, left_eye_y, right_eye_x, right_eye_y,
                      nose_tip_x, nose_tip_y, mouth_x, mouth_y,
                      left_ear_x, left_ear_y, right_ear_x, right_ear_y,
                      yaw, pitch, roll, eye_distance, face_aspect_ratio])
        
        # Insert into database
        placeholders = ','.join(['?' for _ in values])
        query = f"INSERT INTO faces ({','.join(columns)}) VALUES ({placeholders})"
        
        cursor.execute(query, values)
        print(f"  ✓ Stored features for '{name}'")
        print(f"    - 468 facial landmarks (1404 coordinates)")
        print(f"    - Face location: ({face_x:.0f}, {face_y:.0f})")
        print(f"    - Face size: {face_width:.0f}x{face_height:.0f} pixels")
        print(f"    - Orientation: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
        print(f"    - Eye distance: {eye_distance:.1f}px")
    
    conn.commit()
    conn.close()
    face_mesh.close()
    print(f"\n✅ Processed {len(image_files)} images and stored in database!")

def view_database():
    """Display all records in the database"""
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, image_path, face_location_width, face_location_height FROM faces")
    rows = cursor.fetchall()
    
    if not rows:
        print("No records in database yet.")
    else:
        print("\n" + "="*70)
        print("FACE DATABASE RECORDS")
        print("="*70)
        for row in rows:
            print(f"ID: {row[0]} | Name: {row[1]} | Image: {row[2]}")
            print(f"  Face Dimensions: {row[3]:.0f}x{row[4]:.0f} pixels")
            print("-"*70)
        print(f"\nTotal records: {len(rows)}")
        print(f"Total features per record: 1404 landmark coords + 17 computed features = 1421+ features\n")
    
    conn.close()

if __name__ == "__main__":
    print("Face Database Builder")
    print("="*70)
    
    # Create database
    create_database()
    
    # Extract features and store
    extract_and_store_faces()
    
    # View results
    view_database()
