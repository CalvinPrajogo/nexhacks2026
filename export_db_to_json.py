import sqlite3
import json
import os

# Create face_database directory if it doesn't exist
os.makedirs('face_database', exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect('face_database.db')
cursor = conn.cursor()

# Get all faces
cursor.execute('SELECT id, name, image_path FROM faces')
faces = cursor.fetchall()

print(f'Exporting {len(faces)} faces to JSON...')

for face_id, name, image_path in faces:
    # Get all data for this face
    cursor.execute('SELECT * FROM faces WHERE id = ?', (face_id,))
    row = cursor.fetchone()
    
    # Get column names
    cursor.execute('PRAGMA table_info(faces)')
    columns = [col[1] for col in cursor.fetchall()]
    
    # Extract features (skip id, name, image_path)
    features = []
    for i, col in enumerate(columns):
        if col not in ['id', 'name', 'image_path']:
            features.append(float(row[i]) if row[i] is not None else 0.0)
    
    # Create JSON structure
    face_data = {
        "name": name,
        "features": features,
        "image_path": image_path,
        "metadata": {
            "feature_count": len(features),
            "source": "face_database.db"
        }
    }
    
    # Save to individual JSON file
    output_path = os.path.join('face_database', f'{name}.json')
    with open(output_path, 'w') as f:
        json.dump(face_data, f, indent=2)
    
    print(f'✓ Exported {name} → {output_path} ({len(features)} features)')

conn.close()
print(f'\n✅ All faces exported to face_database/ directory')
