"""
Recapture database entries using live camera to ensure consistency
"""
import cv2
import json
import os
from extract_live_face import extract_face_features
from datetime import datetime

def capture_person(name):
    """Capture a person's face for the database"""
    print(f"\n=== Capturing {name} ===")
    print("Press SPACE to capture, ESC to cancel")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Capture Face - Press SPACE', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE
            # Save frame temporarily
            temp_file = f'temp_capture_{name}.jpg'
            cv2.imwrite(temp_file, frame)
            
            # Extract features
            print("Extracting features...")
            features = extract_face_features(temp_file)
            
            if 'error' in features:
                print(f"Error: {features['error']}")
                os.remove(temp_file)
                continue
            
            # Save to database
            db_file = f'face_database/{name}.json'
            data = {
                'name': name,
                'features': list(features.values()),  # Convert dict to list
                'image_path': temp_file,
                'metadata': {
                    'captured_at': datetime.now().isoformat(),
                    'method': 'live_recapture'
                }
            }
            
            with open(db_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved to {db_file}")
            print(f"  Total features: {len(data['features'])}")
            
            cap.release()
            cv2.destroyAllWindows()
            os.remove(temp_file)
            return True
    
    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    print("Database Recapture Tool")
    print("=" * 40)
    
    name = input("Enter person's name (e.g., calvin_prajogo): ").strip()
    if name:
        capture_person(name)
    else:
        print("No name provided")
