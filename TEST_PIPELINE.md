# Complete Pipeline Testing Guide

## Prerequisites
- [ ] Wood Wide AI API key (set as environment variable)
- [ ] All Python dependencies installed
- [ ] Node.js dependencies installed in vision-app/

## Step-by-Step Testing

### 1️⃣ Export Database to JSON Format
```bash
# Export database faces to face_database/ directory
python3 export_db_to_json.py
```
Expected output: 3 JSON files created (calvin_prajogo.json, eden_brunner.json, brinly_richards.json)

---

### 2️⃣ Set Wood Wide API Key
```bash
# Set your Wood Wide API key
export WOODWIDE_API_KEY="your_api_key_here"

# Verify it's set
echo $WOODWIDE_API_KEY
```

---

### 3️⃣ Start Face Matching Server (Port 5001)
```bash
# Start the embedding & matching server
python3 vision-app/src/face_matching_server.py
```
Expected output: 
- "Loaded 3 faces into database"
- "Flask server running on http://0.0.0.0:5001"

Leave this terminal running!

---

### 4️⃣ Train the Embedding Model (One-Time Setup)
In a **new terminal**, run:
```bash
# Train model and generate embeddings for all database faces
curl -X POST http://localhost:5001/full-pipeline
```

Expected process:
1. Training starts → Takes 30-120 seconds
2. Model status polling → "COMPLETE" status
3. Embeddings generated for all 3 faces
4. Embeddings cached to embeddings_cache.pkl

You'll see output like:
```json
{
  "success": true,
  "model_id": "emb_...",
  "status": "COMPLETE",
  "embeddings_generated": 3
}
```

⚠️ **Only need to do this ONCE** unless you add more faces to the database!

---

### 5️⃣ Start Feature Extraction Server (Port 5000)
In a **new terminal**:
```bash
# Start the feature extraction server
python3 feature_server.py
```
Expected output: "Flask server running on http://0.0.0.0:5000"

Leave this terminal running!

---

### 6️⃣ Start React App
In a **new terminal**:
```bash
cd vision-app
npm start
```
Expected: Browser opens at http://localhost:3000

---

### 7️⃣ Test Live Matching
1. **Allow camera access** when prompted
2. **Position yourself in front of camera** (or one of the database faces)
3. **Wait for person detection** (shows bounding box)
4. **Screenshot captures automatically** after 3 consecutive detections
5. **Check console logs** for:
   - Feature extraction success
   - Face matching results
   - Person identification

---

## Testing Match Endpoint Manually

You can test the matching independently:
```bash
# Extract features from a live capture
curl -X POST http://localhost:5001/match-with-woodwide \
  -H "Content-Type: application/json" \
  -d @live_captures/latest_capture.json
```

Expected response:
```json
{
  "success": true,
  "match": {
    "person": "calvin_prajogo",
    "distance": 0.234,
    "confidence": 0.95,
    "method": "woodwide_embeddings"
  }
}
```

---

## Troubleshooting

### "Model not found" error
→ Run Step 4 again to train the model

### "No WOODWIDE_API_KEY" error
→ Set the environment variable in Step 2

### "Database empty" error
→ Run Step 1 to export database

### Camera not working
→ Check browser permissions, try different browser

### No match found
→ Check that database faces are properly loaded
→ Try adjusting lighting/camera angle
→ Check embeddings_cache.pkl exists

---

## File Locations

- Database: `face_database.db`
- Database JSONs: `face_database/*.json`
- Live captures: `live_captures/*.json`
- Embeddings cache: `embeddings_cache.pkl`
- Logs: Terminal outputs from servers

---

## Expected Workflow

```
Video Feed (React)
    ↓
Person Detection (Overshoot SDK)
    ↓
Screenshot Capture (base64)
    ↓
Feature Extraction (Port 5000) → 1,425 features
    ↓
Embedding Conversion (Port 5001) → 128D vector
    ↓
Distance Calculation → Euclidean distance
    ↓
Match Result → Person name + confidence
```
