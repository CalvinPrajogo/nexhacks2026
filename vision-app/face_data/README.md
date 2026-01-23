# Face Data Directory

## Where to Put Your PNG Files

**Place your 3 face photos in the `images/` folder:**
```
face_data/
  └── images/
      ├── person1.png
      ├── person2.png
      └── person3.png
```

## File Naming
- The filename (without .png) will be used as the "name" in the database
- Example: `john_doe.png` → name will be "john_doe"
- Example: `alice.png` → name will be "alice"

## Running the Script
From the project root directory:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the face database builder
python face_database.py
```

## Database Output
The script will create `face_database.db` with 152+ features per face:
- **id**: Auto-incrementing primary key
- **name**: Extracted from filename
- **128 face encoding dimensions** (face_encoding_0 through face_encoding_127)
- **6 face location/size features**: top, right, bottom, left, width, height
- **18 facial landmark coordinates**: chin, eyebrows, nose, eyes, lips (x,y pairs)
