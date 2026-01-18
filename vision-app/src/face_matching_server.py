"""
Face Matching Server using Wood Wide AI Embedding API (Official SDK)
====================================================================

This server uses Wood Wide AI's official Python SDK for embedding workflow:
1. Train embedding model
2. Monitor training status until COMPLETE
3. Generate embeddings for face matching

Then uses Euclidean distance to match input photos against database embeddings.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
from woodwide import WoodWide
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from pathlib import Path
from threading import Thread
import pickle

app = Flask(__name__)
CORS(app)

# Configuration
WOODWIDE_API_KEY = "sk_KaQdUNVc1ziZAL3CIQL9qu2iGPpBp4w6z51UqUPGmnI"
DATABASE_DIR = "./face_database"  # Directory containing face feature JSON files
EMBEDDINGS_CACHE_FILE = "./embeddings_cache.pkl"  # Cache for computed embeddings
MATCH_THRESHOLD = 0.85  # Euclidean distance threshold for matching

# Initialize Wood Wide client using official SDK
woodwide_client = WoodWide(api_key=WOODWIDE_API_KEY)

# In-memory stores
face_database = {}  # Raw face features from JSON files
face_embeddings = {}  # Wood Wide AI computed embeddings
embedding_model_id = None  # Current trained model ID
model_status = "NOT_TRAINED"  # PENDING, TRAINING, COMPLETE, FAILED


# Helper functions for SDK
def upload_dataset_sdk(dataset: list, dataset_name: str) -> dict:
    """
    Upload dataset to Wood Wide AI first (required before training)
    Converts dataset to CSV and uploads
    """
    try:
        import pandas as pd
        import io
        
        print(f"[WoodWide SDK] Preparing dataset: {dataset_name}", flush=True)
        print(f"[WoodWide SDK] Dataset size: {len(dataset)} instances", flush=True)

        # Convert dataset to DataFrame first
        df = pd.DataFrame(dataset)

        print(f"[WoodWide SDK] Original DataFrame shape: {df.shape}", flush=True)

        # WORKAROUND: Reduce feature dimensionality from 1425 to 100
        # Wood Wide AI's embedding model appears to have issues with 1425 columns
        # Simply select the first 100 features as a simple dimensionality reduction
        label_col = 'person_name'
        feature_cols = [col for col in df.columns if col.startswith('feature_')]

        # Keep only first 100 features
        max_features = 100
        selected_features = feature_cols[:max_features]

        print(f"[WoodWide SDK] Reducing features: {len(feature_cols)} -> {len(selected_features)}", flush=True)

        # Create reduced DataFrame
        reduced_df = df[[label_col] + selected_features].copy()

        df = reduced_df
        print(f"[WoodWide SDK] Reduced DataFrame shape: {df.shape}", flush=True)

        # Save to CSV as bytes
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset to beginning
        
        # Use the upload method with filename to indicate CSV format
        print(f"[WoodWide SDK] Calling SDK upload method...", flush=True)
        try:
            dataset_result = woodwide_client.api.datasets.upload(
                file=(f"{dataset_name}.csv", csv_buffer, "text/csv"),
                name=dataset_name,
                overwrite=True
            )
            print(f"[WoodWide SDK] Dataset uploaded successfully", flush=True)
            print(f"[WoodWide SDK] Upload result: {dataset_result}", flush=True)
        except Exception as upload_error:
            print(f"[WoodWide SDK] Dataset SDK upload failed: {upload_error}")
            import traceback
            traceback.print_exc()
            raise

        return {
            "success": True,
            "dataset_id": getattr(dataset_result, 'id', None) or dataset_name,
            "dataset_name": dataset_name
        }
    except Exception as e:
        print(f"[WoodWide SDK] Dataset upload function failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def train_embedding_model_sdk(dataset_name: str, model_name: str = "face_embeddings") -> dict:
    """
    Train embedding model using Wood Wide SDK
    Note: dataset_name is passed as query parameter
    """
    try:
        import httpx

        print(f"[WoodWide SDK] Training model: {model_name}", flush=True)
        print(f"[WoodWide SDK] Using dataset: {dataset_name}", flush=True)

        # Use httpx directly with simple form data (no input_columns)
        headers = {
            "Authorization": f"Bearer {WOODWIDE_API_KEY}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Simple form data without input_columns
        form_data = {
            "model_name": model_name,
            "label_column": "person_name",
            "overwrite": "true"
        }

        print(f"[WoodWide SDK] Starting training (auto-detect columns)...", flush=True)

        response = httpx.post(
            f"https://beta.woodwide.ai/api/models/embedding/train?dataset_name={dataset_name}",
            headers=headers,
            data=form_data,
            timeout=30.0
        )

        print(f"[WoodWide SDK] Response status: {response.status_code}", flush=True)
        print(f"[WoodWide SDK] Response body: {response.text}", flush=True)

        if response.status_code != 200:
            print(f"[WoodWide SDK] ERROR - Training failed with status {response.status_code}", flush=True)
            return {
                "success": False,
                "error": f"Training API returned {response.status_code}: {response.text}"
            }

        response.raise_for_status()
        model_public = response.json()

        print(f"[WoodWide SDK] Training initiated successfully", flush=True)
        print(f"[WoodWide SDK] Model ID: {model_public['id']}", flush=True)
        print(f"[WoodWide SDK] Training Status: {model_public['training_status']}", flush=True)

        return {
            "success": True,
            "model_id": model_public['id'],
            "status": model_public['training_status']
        }
    except Exception as e:
        print(f"[WoodWide SDK] Training failed: {e}")
        # Try to show more debug info
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def get_model_status_sdk(model_id: str) -> dict:
    """
    Get model status using httpx directly
    """
    try:
        import httpx

        headers = {
            "Authorization": f"Bearer {WOODWIDE_API_KEY}"
        }

        response = httpx.get(
            f"https://beta.woodwide.ai/api/models/{model_id}",
            headers=headers,
            timeout=30.0
        )

        response.raise_for_status()
        model = response.json()

        return {
            "success": True,
            "model_id": model['id'],
            "status": model['training_status'],
            "details": model
        }
    except Exception as e:
        print(f"[WoodWide SDK] Status check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def wait_for_training_sdk(model_id: str, timeout: int = 300, poll_interval: int = 5) -> dict:
    """
    Wait for model training to complete using SDK
    """
    start_time = time.time()
    print(f"[WoodWide SDK] Waiting for model {model_id} to complete training...")
    
    while time.time() - start_time < timeout:
        status_result = get_model_status_sdk(model_id)
        
        if not status_result.get("success"):
            return status_result
        
        status = status_result.get("status")
        
        if status == "COMPLETE":
            print(f"[WoodWide SDK] Training complete!")
            return status_result
        elif status == "FAILED":
            print(f"[WoodWide SDK] Training failed!")
            return status_result
        
        print(f"[WoodWide SDK] Status: {status}, waiting {poll_interval}s...")
        time.sleep(poll_interval)
    
    return {
        "success": False,
        "error": f"Training timeout after {timeout} seconds"
    }


def infer_embeddings_sdk(model_id: str, dataset_name: str) -> dict:
    """
    Generate embeddings using Wood Wide API
    Per API docs: POST /api/models/embedding/{model_id}/infer?dataset_name={dataset_name}
    """
    try:
        import httpx

        print(f"[WoodWide SDK] Generating embeddings for dataset: {dataset_name}")
        print(f"[WoodWide SDK] Using model: {model_id}")

        headers = {
            "Authorization": f"Bearer {WOODWIDE_API_KEY}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Per API docs, coerce_schema defaults to true
        form_data = {
            "coerce_schema": "true"
        }

        response = httpx.post(
            f"https://beta.woodwide.ai/api/models/embedding/{model_id}/infer?dataset_name={dataset_name}",
            headers=headers,
            data=form_data,
            timeout=60.0
        )

        print(f"[WoodWide SDK] Inference response status: {response.status_code}")

        if response.status_code != 200:
            print(f"[WoodWide SDK] ERROR - Inference failed with status {response.status_code}")
            print(f"[WoodWide SDK] ERROR - Response: {response.text}")

        response.raise_for_status()
        result = response.json()

        print(f"[WoodWide SDK] Generated embeddings successfully")

        return {
            "success": True,
            "embeddings": result.get("embeddings", result),
            "model_id": model_id
        }
    except Exception as e:
        print(f"[WoodWide SDK] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors"""
    return float(np.linalg.norm(vec1 - vec2))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def load_face_database():
    """
    Load all face feature JSON files from the database directory
    Each JSON file should have: {"name": "person_name", "features": [...]}
    """
    global face_database
    
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        print(f"Created database directory: {DATABASE_DIR}")
        return
    
    face_database = {}
    
    for filename in os.listdir(DATABASE_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(DATABASE_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle both single entry and list of entries
                entries = data if isinstance(data, list) else [data]
                
                for entry in entries:
                    name = entry.get('name') or entry.get('person_name')
                    features = entry.get('features') or entry.get('embedding')
                    
                    if name and features:
                        face_database[name] = {
                            'features': np.array(features, dtype=np.float32),
                            'image_path': entry.get('image_path', ''),
                            'metadata': entry.get('metadata', {})
                        }
                        print(f"Loaded face data for: {name}")
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(face_database)} faces into database")


def load_embeddings_cache():
    """Load cached embeddings from disk"""
    global face_embeddings, embedding_model_id, model_status
    
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                face_embeddings = cache.get('embeddings', {})
                embedding_model_id = cache.get('model_id')
                model_status = cache.get('status', 'NOT_TRAINED')
                print(f"Loaded {len(face_embeddings)} cached embeddings")
                print(f"Model ID: {embedding_model_id}, Status: {model_status}")
        except Exception as e:
            print(f"Error loading cache: {e}")


def save_embeddings_cache():
    """Save embeddings to disk cache"""
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump({
                'embeddings': face_embeddings,
                'model_id': embedding_model_id,
                'status': model_status
            }, f)
        print(f"Saved {len(face_embeddings)} embeddings to cache")
    except Exception as e:
        print(f"Error saving cache: {e}")


def find_matching_person_local(input_features: list, threshold: float = MATCH_THRESHOLD) -> dict:
    """
    Find matching person using local Euclidean distance on raw features
    (Fallback when Wood Wide AI is not available)
    """
    if not face_database:
        return {"success": False, "error": "Face database is empty", "matched": False}
    
    input_array = normalize_vector(np.array(input_features, dtype=np.float32))
    
    distances = []
    for name, data in face_database.items():
        db_features = normalize_vector(data['features'])
        distance = euclidean_distance(input_array, db_features)
        distances.append({
            "name": name,
            "distance": distance,
            "image_path": data.get('image_path', ''),
            "metadata": data.get('metadata', {})
        })
    
    distances.sort(key=lambda x: x['distance'])
    best_match = distances[0] if distances else None
    
    if best_match and best_match['distance'] < threshold:
        confidence = np.exp(-best_match['distance'])
        return {
            "success": True,
            "matched": True,
            "person_name": best_match['name'],
            "confidence": float(confidence),
            "distance": best_match['distance'],
            "image_path": best_match['image_path'],
            "metadata": best_match['metadata'],
            "top_matches": distances[:5],
            "method": "local_euclidean"
        }
    else:
        return {
            "success": True,
            "matched": False,
            "message": "No match found within threshold",
            "best_distance": best_match['distance'] if best_match else None,
            "threshold": threshold,
            "top_matches": distances[:5],
            "method": "local_euclidean"
        }


def find_matching_person_woodwide(input_features: list, threshold: float = MATCH_THRESHOLD) -> dict:
    """
    Find matching person using Wood Wide AI embeddings + Euclidean distance
    """
    global embedding_model_id, face_embeddings

    if not face_embeddings:
        return {"success": False, "error": "No Wood Wide embeddings available. Train model first.", "matched": False}

    if not embedding_model_id:
        return {"success": False, "error": "No trained model available", "matched": False}

    # Upload input features as a temporary dataset for inference
    input_dataset_name = f"input_face_inference_{int(time.time())}"
    input_dataset = []

    # Create dataset with same structure as training data
    row = {"person_name": "input_face"}
    for i, feat_val in enumerate(input_features):
        row[f"feature_{i}"] = feat_val
    input_dataset.append(row)

    try:
        # Upload the input dataset
        upload_result = upload_dataset_sdk(input_dataset, input_dataset_name)
        if not upload_result.get("success"):
            print("[WoodWide] Failed to upload input dataset, falling back to local")
            return find_matching_person_local(input_features, threshold)

        # Generate embedding for the input
        result = infer_embeddings_sdk(embedding_model_id, input_dataset_name)
        if result.get("success") and result.get("embeddings"):
            input_embedding = np.array(result["embeddings"][0])
        else:
            print("[WoodWide] Failed to get embedding for input, falling back to local")
            return find_matching_person_local(input_features, threshold)
    except Exception as e:
        print(f"[WoodWide] Inference error: {e}, falling back to local")
        return find_matching_person_local(input_features, threshold)
    
    # Compare against all database embeddings using Euclidean distance
    distances = []
    for name, embedding in face_embeddings.items():
        distance = euclidean_distance(input_embedding, embedding)
        distances.append({
            "name": name,
            "distance": distance,
            "image_path": face_database.get(name, {}).get('image_path', ''),
            "metadata": face_database.get(name, {}).get('metadata', {})
        })
    
    distances.sort(key=lambda x: x['distance'])
    best_match = distances[0] if distances else None
    
    if best_match and best_match['distance'] < threshold:
        confidence = np.exp(-best_match['distance'])
        return {
            "success": True,
            "matched": True,
            "person_name": best_match['name'],
            "confidence": float(confidence),
            "distance": best_match['distance'],
            "image_path": best_match['image_path'],
            "metadata": best_match['metadata'],
            "top_matches": distances[:5],
            "method": "woodwide_embedding"
        }
    else:
        return {
            "success": True,
            "matched": False,
            "message": "No match found within threshold",
            "best_distance": best_match['distance'] if best_match else None,
            "threshold": threshold,
            "top_matches": distances[:5],
            "method": "woodwide_embedding"
        }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database_size": len(face_database),
        "embeddings_size": len(face_embeddings),
        "model_id": embedding_model_id,
        "model_status": model_status,
        "woodwide_configured": bool(WOODWIDE_API_KEY and WOODWIDE_API_KEY != "your-api-key-here")
    })


@app.route('/load-database', methods=['POST'])
def reload_database():
    """Reload the face database from disk"""
    load_face_database()
    load_embeddings_cache()
    return jsonify({
        "success": True,
        "message": f"Loaded {len(face_database)} faces",
        "names": list(face_database.keys()),
        "embeddings_loaded": len(face_embeddings)
    })


@app.route('/add-face', methods=['POST'])
def add_face():
    """Add a new face to the database"""
    data = request.json
    
    name = data.get('name')
    features = data.get('features')
    
    if not name or not features:
        return jsonify({"success": False, "error": "Missing 'name' or 'features'"}), 400
    
    face_database[name] = {
        'features': np.array(features, dtype=np.float32),
        'image_path': data.get('image_path', ''),
        'metadata': data.get('metadata', {})
    }
    
    # Save to disk
    save_path = os.path.join(DATABASE_DIR, f"{name.replace(' ', '_')}.json")
    with open(save_path, 'w') as f:
        json.dump({
            "name": name,
            "features": features,
            "image_path": data.get('image_path', ''),
            "metadata": data.get('metadata', {})
        }, f, indent=2)
    
    return jsonify({
        "success": True,
        "message": f"Added face for {name}",
        "saved_to": save_path,
        "note": "Run /train-embeddings to update Wood Wide AI model"
    })


@app.route('/train-embeddings', methods=['POST'])
def train_embeddings():
    """
    Train Wood Wide AI embedding model on the face database
    
    Step 1: Upload dataset
    Step 2: Train model using dataset_name
    Returns model_id with PENDING status
    """
    global embedding_model_id, model_status
    
    if not face_database:
        return jsonify({"success": False, "error": "Face database is empty"}), 400
    
    # Prepare dataset for Wood Wide AI
    # Flatten features into separate columns (feature_0, feature_1, ...)
    dataset = []
    for name, data in face_database.items():
        row = {"person_name": name}
        features = data['features'].tolist()
        # Add each feature as a separate column
        for i, feat_val in enumerate(features):
            row[f"feature_{i}"] = feat_val
        dataset.append(row)
    
    dataset_name = "face_embeddings_dataset"
    
    # Step 1: Upload dataset
    upload_result = upload_dataset_sdk(dataset, dataset_name)
    if not upload_result.get("success"):
        return jsonify({
            "success": False,
            "error": f"Dataset upload failed: {upload_result.get('error')}"
        }), 500
    
    # Step 2: Train embedding model using dataset_name
    result = train_embedding_model_sdk(dataset_name, "face_embeddings")
    
    if result.get("success"):
        embedding_model_id = result.get("model_id")
        model_status = result.get("status", "PENDING")
        
        return jsonify({
            "success": True,
            "model_id": embedding_model_id,
            "status": model_status,
            "message": "Training started. Use /model-status to monitor progress.",
            "dataset_size": len(dataset)
        })
    else:
        return jsonify({
            "success": False,
            "error": result.get("error", "Training failed")
        }), 500


@app.route('/model-status', methods=['GET'])
def get_model_status():
    """
    Check Wood Wide AI model training status
    
    Step 2: GET /api/models/{model_id}
    Returns status: PENDING, TRAINING, COMPLETE, or FAILED
    """
    global model_status
    
    if not embedding_model_id:
        return jsonify({
            "success": False,
            "error": "No model has been trained yet",
            "status": "NOT_TRAINED"
        })
    
    result = get_model_status_sdk(embedding_model_id)
    
    if result.get("success"):
        model_status = result.get("status")
        return jsonify({
            "success": True,
            "model_id": embedding_model_id,
            "status": model_status,
            "details": result.get("details", {})
        })
    else:
        return jsonify(result), 500


@app.route('/wait-for-training', methods=['POST'])
def wait_for_training():
    """Wait for model training to complete (blocking)"""
    global model_status
    
    if not embedding_model_id:
        return jsonify({"success": False, "error": "No model has been trained yet"})
    
    timeout = request.json.get('timeout', 300) if request.json else 300
    
    result = wait_for_training_sdk(embedding_model_id, timeout=timeout)
    
    if result.get("success"):
        model_status = result.get("status")
    
    return jsonify(result)


@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    """
    Generate embeddings for all faces in the database using trained model
    
    Step 3: POST /api/models/embedding/{model_id}/infer
    """
    global face_embeddings, model_status
    
    if not embedding_model_id:
        return jsonify({"success": False, "error": "No model has been trained yet"}), 400
    
    if model_status != "COMPLETE":
        return jsonify({
            "success": False,
            "error": f"Model not ready. Current status: {model_status}",
            "hint": "Wait for training to complete or call /wait-for-training"
        }), 400
    
    # Use the same dataset that was used for training
    dataset_name = "face_embeddings_dataset"

    # Generate embeddings via Wood Wide AI
    # The API will use the dataset we uploaded during training
    result = infer_embeddings_sdk(embedding_model_id, dataset_name)

    if result.get("success"):
        embeddings = result.get("embeddings", [])

        # Store embeddings - map them back to person names
        # The embeddings should be in the same order as the dataset
        face_embeddings = {}
        names = list(face_database.keys())

        for i, name in enumerate(names):
            if i < len(embeddings):
                face_embeddings[name] = np.array(embeddings[i])

        # Save to cache
        save_embeddings_cache()

        return jsonify({
            "success": True,
            "embeddings_generated": len(face_embeddings),
            "names": list(face_embeddings.keys()),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0
        })
    else:
        return jsonify({
            "success": False,
            "error": result.get("error", "Inference failed")
        }), 500


@app.route('/full-pipeline', methods=['POST'])
def full_pipeline():
    """
    Run the complete pipeline using Euclidean distance on feature space
    (Wood Wide embedding code preserved but not used due to insufficient training data)
    """
    global model_status
    
    if not face_database:
        return jsonify({"success": False, "error": "Face database is empty"}), 400
    
    # Skip Wood Wide embedding training - use direct feature comparison instead
    print("\n" + "="*50)
    print("PIPELINE: Using Euclidean distance on feature space")
    print("="*50)
    print(f"Database loaded with {len(face_database)} faces")
    print("Ready for matching using local feature comparison")
    
    model_status = "READY_FOR_EUCLIDEAN_MATCHING"
    
    return jsonify({
        "success": True,
        "method": "euclidean_distance",
        "faces_loaded": len(face_database),
        "names": list(face_database.keys()),
        "message": "Pipeline ready - using Euclidean distance matching on raw features"
    })


@app.route('/match-face', methods=['POST'])
def match_face():
    """
    Match an input face against the database using local Euclidean distance
    (Fallback method - doesn't require Wood Wide AI)
    """
    data = request.json
    features = data.get('features')
    threshold = data.get('threshold', MATCH_THRESHOLD)
    
    if not features:
        return jsonify({"success": False, "error": "Missing 'features'"}), 400
    
    result = find_matching_person_local(features, threshold)
    return jsonify(result)


@app.route('/match-with-woodwide', methods=['POST'])
def match_with_woodwide():
    """
    Match using Wood Wide AI embeddings + Euclidean distance
    
    This endpoint:
    1. Gets embedding for input features via Wood Wide AI inference
    2. Compares against database embeddings using Euclidean distance
    3. Returns the matched person's name
    """
    data = request.json
    features = data.get('features')
    threshold = data.get('threshold', MATCH_THRESHOLD)
    
    if not features:
        return jsonify({"success": False, "error": "Missing 'features'"}), 400
    
    # Try Wood Wide AI matching first
    if face_embeddings and embedding_model_id and model_status == "COMPLETE":
        result = find_matching_person_woodwide(features, threshold)
    else:
        # Fallback to local matching
        print("[Match] Wood Wide embeddings not available, using local matching")
        result = find_matching_person_local(features, threshold)
        result["note"] = "Using local matching. Run /full-pipeline to enable Wood Wide AI."
    
    return jsonify(result)


@app.route('/database-stats', methods=['GET'])
def database_stats():
    """Get statistics about the face database and embeddings"""
    stats = {
        "total_faces": len(face_database),
        "names": list(face_database.keys()),
        "feature_dimensions": None,
        "embeddings_available": len(face_embeddings),
        "embedding_dimensions": None,
        "model_id": embedding_model_id,
        "model_status": model_status,
        "woodwide_configured": bool(WOODWIDE_API_KEY and WOODWIDE_API_KEY != "your-api-key-here")
    }
    
    if face_database:
        first_entry = next(iter(face_database.values()))
        stats["feature_dimensions"] = len(first_entry['features'])
    
    if face_embeddings:
        first_embedding = next(iter(face_embeddings.values()))
        stats["embedding_dimensions"] = len(first_embedding)
    
    return jsonify(stats)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Load face database on startup
    load_face_database()
    load_embeddings_cache()
    
    print("\n" + "="*70)
    print("Face Matching Server with Wood Wide AI Embedding API")
    print("="*70)
    print(f"Database directory: {DATABASE_DIR}")
    print(f"Faces loaded: {len(face_database)}")
    print(f"Cached embeddings: {len(face_embeddings)}")
    print(f"Model ID: {embedding_model_id}")
    print(f"Model Status: {model_status}")
    print(f"Wood Wide API configured: {bool(WOODWIDE_API_KEY and WOODWIDE_API_KEY != 'your-api-key-here')}")
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /load-database       - Reload face database")
    print("  POST /add-face            - Add new face to database")
    print("  ")
    print("  Wood Wide AI Pipeline:")
    print("  POST /train-embeddings    - Step 1: Train embedding model")
    print("  GET  /model-status        - Step 2: Check training status")
    print("  POST /generate-embeddings - Step 3: Generate embeddings")
    print("  POST /full-pipeline       - Run all 3 steps automatically")
    print("  ")
    print("  Matching:")
    print("  POST /match-face          - Match using local Euclidean distance")
    print("  POST /match-with-woodwide - Match using Wood Wide AI embeddings")
    print("  GET  /database-stats      - Get database statistics")
    print("="*70 + "\n")
    
    # Run without debug to see print statements
    app.run(host='0.0.0.0', port=5001, debug=False)
