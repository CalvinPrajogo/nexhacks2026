import face_recognition
import sqlite3
import os
import numpy as np
from PIL import Image

def create_database():
    """Create SQLite database with face features table"""
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    
    # Create table with id, name, and 128 face encoding features + additional features
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            face_encoding_0 REAL, face_encoding_1 REAL, face_encoding_2 REAL, face_encoding_3 REAL,
            face_encoding_4 REAL, face_encoding_5 REAL, face_encoding_6 REAL, face_encoding_7 REAL,
            face_encoding_8 REAL, face_encoding_9 REAL, face_encoding_10 REAL, face_encoding_11 REAL,
            face_encoding_12 REAL, face_encoding_13 REAL, face_encoding_14 REAL, face_encoding_15 REAL,
            face_encoding_16 REAL, face_encoding_17 REAL, face_encoding_18 REAL, face_encoding_19 REAL,
            face_encoding_20 REAL, face_encoding_21 REAL, face_encoding_22 REAL, face_encoding_23 REAL,
            face_encoding_24 REAL, face_encoding_25 REAL, face_encoding_26 REAL, face_encoding_27 REAL,
            face_encoding_28 REAL, face_encoding_29 REAL, face_encoding_30 REAL, face_encoding_31 REAL,
            face_encoding_32 REAL, face_encoding_33 REAL, face_encoding_34 REAL, face_encoding_35 REAL,
            face_encoding_36 REAL, face_encoding_37 REAL, face_encoding_38 REAL, face_encoding_39 REAL,
            face_encoding_40 REAL, face_encoding_41 REAL, face_encoding_42 REAL, face_encoding_43 REAL,
            face_encoding_44 REAL, face_encoding_45 REAL, face_encoding_46 REAL, face_encoding_47 REAL,
            face_encoding_48 REAL, face_encoding_49 REAL, face_encoding_50 REAL, face_encoding_51 REAL,
            face_encoding_52 REAL, face_encoding_53 REAL, face_encoding_54 REAL, face_encoding_55 REAL,
            face_encoding_56 REAL, face_encoding_57 REAL, face_encoding_58 REAL, face_encoding_59 REAL,
            face_encoding_60 REAL, face_encoding_61 REAL, face_encoding_62 REAL, face_encoding_63 REAL,
            face_encoding_64 REAL, face_encoding_65 REAL, face_encoding_66 REAL, face_encoding_67 REAL,
            face_encoding_68 REAL, face_encoding_69 REAL, face_encoding_70 REAL, face_encoding_71 REAL,
            face_encoding_72 REAL, face_encoding_73 REAL, face_encoding_74 REAL, face_encoding_75 REAL,
            face_encoding_76 REAL, face_encoding_77 REAL, face_encoding_78 REAL, face_encoding_79 REAL,
            face_encoding_80 REAL, face_encoding_81 REAL, face_encoding_82 REAL, face_encoding_83 REAL,
            face_encoding_84 REAL, face_encoding_85 REAL, face_encoding_86 REAL, face_encoding_87 REAL,
            face_encoding_88 REAL, face_encoding_89 REAL, face_encoding_90 REAL, face_encoding_91 REAL,
            face_encoding_92 REAL, face_encoding_93 REAL, face_encoding_94 REAL, face_encoding_95 REAL,
            face_encoding_96 REAL, face_encoding_97 REAL, face_encoding_98 REAL, face_encoding_99 REAL,
            face_encoding_100 REAL, face_encoding_101 REAL, face_encoding_102 REAL, face_encoding_103 REAL,
            face_encoding_104 REAL, face_encoding_105 REAL, face_encoding_106 REAL, face_encoding_107 REAL,
            face_encoding_108 REAL, face_encoding_109 REAL, face_encoding_110 REAL, face_encoding_111 REAL,
            face_encoding_112 REAL, face_encoding_113 REAL, face_encoding_114 REAL, face_encoding_115 REAL,
            face_encoding_116 REAL, face_encoding_117 REAL, face_encoding_118 REAL, face_encoding_119 REAL,
            face_encoding_120 REAL, face_encoding_121 REAL, face_encoding_122 REAL, face_encoding_123 REAL,
            face_encoding_124 REAL, face_encoding_125 REAL, face_encoding_126 REAL, face_encoding_127 REAL,
            face_location_top INTEGER,
            face_location_right INTEGER,
            face_location_bottom INTEGER,
            face_location_left INTEGER,
            face_width INTEGER,
            face_height INTEGER,
            chin_x REAL, chin_y REAL,
            left_eyebrow_x REAL, left_eyebrow_y REAL,
            right_eyebrow_x REAL, right_eyebrow_y REAL,
            nose_bridge_x REAL, nose_bridge_y REAL,
            nose_tip_x REAL, nose_tip_y REAL,
            left_eye_x REAL, left_eye_y REAL,
            right_eye_x REAL, right_eye_y REAL,
            top_lip_x REAL, top_lip_y REAL,
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
        image = face_recognition.load_image_file(image_path)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
        
        if len(face_encodings) == 0:
            print(f"  ⚠️  No face detected in {image_file}")
            continue
        
        if len(face_encodings) > 1:
            print(f"  ⚠️  Multiple faces detected in {image_file}, using the first one")
        
        # Get the first face
        face_encoding = face_encodings[0]
        face_location = face_locations[0]
        face_landmarks = face_landmarks_list[0] if face_landmarks_list else {}
        
        # Extract name from filename (remove .png extension)
        name = os.path.splitext(image_file)[0]
        
        # Calculate face dimensions
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        # Extract landmark averages (center points of each feature)
        def get_avg_landmark(landmarks, key):
            if key in landmarks:
                points = landmarks[key]
                avg_x = sum(p[0] for p in points) / len(points)
                avg_y = sum(p[1] for p in points) / len(points)
                return avg_x, avg_y
            return None, None
        
        chin_x, chin_y = get_avg_landmark(face_landmarks, 'chin')
        left_eyebrow_x, left_eyebrow_y = get_avg_landmark(face_landmarks, 'left_eyebrow')
        right_eyebrow_x, right_eyebrow_y = get_avg_landmark(face_landmarks, 'right_eyebrow')
        nose_bridge_x, nose_bridge_y = get_avg_landmark(face_landmarks, 'nose_bridge')
        nose_tip_x, nose_tip_y = get_avg_landmark(face_landmarks, 'nose_tip')
        left_eye_x, left_eye_y = get_avg_landmark(face_landmarks, 'left_eye')
        right_eye_x, right_eye_y = get_avg_landmark(face_landmarks, 'right_eye')
        top_lip_x, top_lip_y = get_avg_landmark(face_landmarks, 'top_lip')
        bottom_lip_x, bottom_lip_y = get_avg_landmark(face_landmarks, 'bottom_lip')
        
        # Prepare data for insertion
        encoding_values = [float(val) for val in face_encoding]
        
        # Build the INSERT query
        columns = ['name', 'image_path']
        values = [name, image_path]
        
        # Add face encodings
        for i in range(128):
            columns.append(f'face_encoding_{i}')
            values.append(encoding_values[i])
        
        # Add face location and dimensions
        columns.extend(['face_location_top', 'face_location_right', 'face_location_bottom', 
                       'face_location_left', 'face_width', 'face_height'])
        values.extend([top, right, bottom, left, face_width, face_height])
        
        # Add landmarks
        columns.extend(['chin_x', 'chin_y', 'left_eyebrow_x', 'left_eyebrow_y',
                       'right_eyebrow_x', 'right_eyebrow_y', 'nose_bridge_x', 'nose_bridge_y',
                       'nose_tip_x', 'nose_tip_y', 'left_eye_x', 'left_eye_y',
                       'right_eye_x', 'right_eye_y', 'top_lip_x', 'top_lip_y',
                       'bottom_lip_x', 'bottom_lip_y'])
        values.extend([chin_x, chin_y, left_eyebrow_x, left_eyebrow_y,
                      right_eyebrow_x, right_eyebrow_y, nose_bridge_x, nose_bridge_y,
                      nose_tip_x, nose_tip_y, left_eye_x, left_eye_y,
                      right_eye_x, right_eye_y, top_lip_x, top_lip_y,
                      bottom_lip_x, bottom_lip_y])
        
        # Insert into database
        placeholders = ','.join(['?' for _ in values])
        query = f"INSERT INTO faces ({','.join(columns)}) VALUES ({placeholders})"
        
        cursor.execute(query, values)
        print(f"  ✓ Stored features for '{name}'")
        print(f"    - Face encoding: 128 dimensions")
        print(f"    - Face location: ({top}, {right}, {bottom}, {left})")
        print(f"    - Face size: {face_width}x{face_height} pixels")
        print(f"    - Landmarks: 9 facial feature groups")
    
    conn.commit()
    conn.close()
    print(f"\n✅ Processed {len(image_files)} images and stored in database!")

def view_database():
    """Display all records in the database"""
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, image_path, face_width, face_height FROM faces")
    rows = cursor.fetchall()
    
    if not rows:
        print("No records in database yet.")
    else:
        print("\n" + "="*70)
        print("FACE DATABASE RECORDS")
        print("="*70)
        for row in rows:
            print(f"ID: {row[0]} | Name: {row[1]} | Image: {row[2]}")
            print(f"  Face Dimensions: {row[3]}x{row[4]} pixels")
            print("-"*70)
        print(f"\nTotal records: {len(rows)}")
        print(f"Total features per record: 128 encodings + 6 location/size + 18 landmarks = 152 features\n")
    
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
