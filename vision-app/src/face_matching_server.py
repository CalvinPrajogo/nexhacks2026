"""
Face Matching Server using Wood Wide AI Embedding API
=====================================================

This server uses Wood Wide AI's embedding model workflow:
1. Train embedding model via POST /api/models/embedding/train
2. Monitor training via GET /api/models/{model_id} until COMPLETE
3. Generate embeddings via POST /api/models/embedding/{model_id}/infer

Then uses Euclidean distance to match input photos against database embeddings.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import requests
import time
from pathlib import Path
from threading import Thread
import pickle

app = Flask(__name__)
CORS(app)

# Configuration
WOODWIDE_API_BASE = "https://api.woodwide.ai"  # Wood Wide AI API base URL
WOODWIDE_API_KEY = os.environ.get("WOODWIDE_API_KEY", "your-api-key-here")
DATABASE_DIR = "./face_database"  # Directory containing face feature JSON files
EMBEDDINGS_CACHE_FILE = "./embeddings_cache.pkl"  # Cache for computed embeddings
MATCH_THRESHOLD = 0.85  # Euclidean distance threshold for matching

# In-memory stores
face_database = {}  # Raw face features from JSON files
face_embeddings = {}  # Wood Wide AI computed embeddings
embedding_model_id = None  # Current trained model ID
model_status = "NOT_TRAINED"  # PENDING, TRAINING, COMPLETE, FAILED


class WoodWideEmbeddingClient:
    """
    Client for Wood Wide AI Embedding API
    
    Workflow:
    1. Train embedding model via POST /api/models/embedding/train
    2. Monitor training via GET /api/models/{model_id} until COMPLETE
    3. Generate embeddings via POST /api/models/embedding/{model_id}/infer
    """
    
    def __init__(self, api_base: str, api_key: str):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model_id = None
        self.model_status = "NOT_TRAINED"
    
    def train_embedding_model(self, dataset: list, dataset_name: str = "face_features") -> dict:
        """
        Step 1: Train an embedding model
        POST /api/models/embedding/train
        
        Args:
            dataset: List of data instances (face features with metadata)
            dataset_name: Name for the dataset
            
        Returns:
            dict with model_id and status (initially PENDING)
        """
        endpoint = f"{self.api_base}/api/models/embedding/train"
        
        payload = {
            "dataset_name": dataset_name,
            "dataset": dataset,
            "config": {
                "description": "Face feature embeddings for person identification",
                "embedding_dimensions": 128,  # Output embedding size
                "training_epochs": 100
            }
        }
        
        print(f"[WoodWide] Training embedding model...")
        print(f"[WoodWide] Endpoint: {endpoint}")
        print(f"[WoodWide] Dataset size: {len(dataset)} instances")
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            self.model_id = result.get("model_id")
            self.model_status = result.get("status", "PENDING")
            
            print(f"[WoodWide] Model ID: {self.model_id}")
            print(f"[WoodWide] Initial Status: {self.model_status}")
            
            return {
                "success": True,
                "model_id": self.model_id,
                "status": self.model_status
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[WoodWide] Training request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_status(self, model_id: str = None) -> dict:
        """
        Step 2: Monitor training status
        GET /api/models/{model_id}
        
        Args:
            model_id: The model ID to check (uses stored ID if not provided)
            
        Returns:
            dict with current status (PENDING, TRAINING, COMPLETE, FAILED)
        """
        model_id = model_id or self.model_id
        if not model_id:
            return {"success": False, "error": "No model ID available"}
        
        endpoint = f"{self.api_base}/api/models/{model_id}"
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            self.model_status = result.get("status", "UNKNOWN")
            
            print(f"[WoodWide] Model {model_id} status: {self.model_status}")
            
            return {
                "success": True,
                "model_id": model_id,
                "status": self.model_status,
                "details": result
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[WoodWide] Status check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def wait_for_training(self, model_id: str = None, timeout: int = 300, poll_interval: int = 5) -> dict:
        """
        Wait for model training to complete
        
        Args:
            model_id: The model ID to monitor
            timeout: Maximum seconds to wait
            poll_interval: Seconds between status checks
            
        Returns:
            Final status dict
        """
        model_id = model_id or self.model_id
        start_time = time.time()
        
        print(f"[WoodWide] Waiting for model {model_id} to complete training...")
        
        while time.time() - start_time < timeout:
            status_result = self.get_model_status(model_id)
            
            if not status_result.get("success"):
                return status_result
            
            status = status_result.get("status")
            
            if status == "COMPLETE":
                print(f"[WoodWide] Training complete!")
                return status_result
            elif status == "FAILED":
                print(f"[WoodWide] Training failed!")
                return status_result
            
            print(f"[WoodWide] Status: {status}, waiting {poll_interval}s...")
            time.sleep(poll_interval)
        
        return {
            "success": False,
            "error": f"Training timeout after {timeout} seconds",
            "last_status": self.model_status
        }
    
    def infer_embeddings(self, model_id: str, instances: list, coerce_schema: bool = True) -> dict:
        """
        Step 3: Generate embeddings for data instances
        POST /api/models/embedding/{model_id}/infer
        
        Args:
            model_id: The trained model ID
            instances: List of data instances to embed
            coerce_schema: Whether to coerce schema (default True)
            
        Returns:
            dict with embedding vectors for each instance
        """
        model_id = model_id or self.model_id
        if not model_id:
            return {"success": False, "error": "No model ID available"}
        
        endpoint = f"{self.api_base}/api/models/embedding/{model_id}/infer"
        
        payload = {
            "instances": instances,
            "coerce_schema": coerce_schema
        }
        
        print(f"[WoodWide] Generating embeddings for {len(instances)} instances...")
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            embeddings = result.get("embeddings", [])
            print(f"[WoodWide] Generated {len(embeddings)} embeddings")
            
            return {
                "success": True,
                "embeddings": embeddings,
                "model_id": model_id
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[WoodWide] Inference failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def infer_single(self, model_id: str, instance: dict) -> np.ndarray:
        """
        Generate embedding for a single instance
        
        Args:
            model_id: The trained model ID
            instance: Single data instance
            
        Returns:
            numpy array of embedding vector, or None on failure
        """
        result = self.infer_embeddings(model_id, [instance])
        
        if result.get("success") and result.get("embeddings"):
            return np.array(result["embeddings"][0])
        
        return None


# Initialize Wood Wide AI client
woodwide_client = WoodWideEmbeddingClient(WOODWIDE_API_BASE, WOODWIDE_API_KEY)


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
    
    # Get embedding for input features using Wood Wide AI
    input_instance = {"features": input_features}
    input_embedding = woodwide_client.infer_single(embedding_model_id, input_instance)
    
    if input_embedding is None:
        print("[WoodWide] Failed to get embedding for input, falling back to local")
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
    
    Step 1: POST /api/models/embedding/train
    Returns model_id with PENDING status
    """
    global embedding_model_id, model_status
    
    if not face_database:
        return jsonify({"success": False, "error": "Face database is empty"}), 400
    
    # Prepare dataset for Wood Wide AI
    dataset = []
    for name, data in face_database.items():
        dataset.append({
            "person_name": name,
            "features": data['features'].tolist(),
            "image_path": data.get('image_path', '')
        })
    
    # Train embedding model
    result = woodwide_client.train_embedding_model(dataset, "face_embeddings")
    
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
    
    result = woodwide_client.get_model_status(embedding_model_id)
    
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
    
    result = woodwide_client.wait_for_training(embedding_model_id, timeout=timeout)
    
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
    
    # Prepare instances for inference
    instances = []
    names = []
    for name, data in face_database.items():
        instances.append({
            "person_name": name,
            "features": data['features'].tolist()
        })
        names.append(name)
    
    # Generate embeddings via Wood Wide AI
    result = woodwide_client.infer_embeddings(embedding_model_id, instances)
    
    if result.get("success"):
        embeddings = result.get("embeddings", [])
        
        # Store embeddings
        face_embeddings = {}
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
    Run the complete Wood Wide AI pipeline:
    1. Train embedding model
    2. Wait for training to complete
    3. Generate embeddings for all faces
    """
    global embedding_model_id, model_status, face_embeddings
    
    if not face_database:
        return jsonify({"success": False, "error": "Face database is empty"}), 400
    
    # Step 1: Train
    print("\n" + "="*50)
    print("STEP 1: Training embedding model...")
    print("="*50)
    
    dataset = []
    for name, data in face_database.items():
        dataset.append({
            "person_name": name,
            "features": data['features'].tolist()
        })
    
    train_result = woodwide_client.train_embedding_model(dataset)
    
    if not train_result.get("success"):
        return jsonify({
            "success": False,
            "step": "train",
            "error": train_result.get("error")
        }), 500
    
    embedding_model_id = train_result.get("model_id")
    
    # Step 2: Wait for training
    print("\n" + "="*50)
    print("STEP 2: Waiting for training to complete...")
    print("="*50)
    
    timeout = request.json.get('timeout', 300) if request.json else 300
    wait_result = woodwide_client.wait_for_training(embedding_model_id, timeout=timeout)
    
    if not wait_result.get("success") or wait_result.get("status") != "COMPLETE":
        return jsonify({
            "success": False,
            "step": "wait",
            "error": wait_result.get("error", "Training did not complete"),
            "status": wait_result.get("status")
        }), 500
    
    model_status = "COMPLETE"
    
    # Step 3: Generate embeddings
    print("\n" + "="*50)
    print("STEP 3: Generating embeddings...")
    print("="*50)
    
    instances = []
    names = []
    for name, data in face_database.items():
        instances.append({
            "person_name": name,
            "features": data['features'].tolist()
        })
        names.append(name)
    
    infer_result = woodwide_client.infer_embeddings(embedding_model_id, instances)
    
    if not infer_result.get("success"):
        return jsonify({
            "success": False,
            "step": "infer",
            "error": infer_result.get("error")
        }), 500
    
    embeddings = infer_result.get("embeddings", [])
    face_embeddings = {}
    for i, name in enumerate(names):
        if i < len(embeddings):
            face_embeddings[name] = np.array(embeddings[i])
    
    save_embeddings_cache()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    
    return jsonify({
        "success": True,
        "model_id": embedding_model_id,
        "embeddings_generated": len(face_embeddings),
        "names": list(face_embeddings.keys()),
        "message": "Full pipeline completed successfully!"
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
    
    app.run(host='0.0.0.0', port=5001, debug=True)