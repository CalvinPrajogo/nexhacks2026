import { useEffect, useRef, useState } from 'react';
import { RealtimeVision } from '@overshoot/sdk';

function VisionApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const visionRef = useRef(null);
  
  // Original state
  const [result, setResult] = useState('');  // <-- This was missing!
  
  // State for face matching
  const [matchResult, setMatchResult] = useState(null);
  const [modelStatus, setModelStatus] = useState('NOT_TRAINED');
  const [isProcessing, setIsProcessing] = useState(false);

  // Server endpoints
  const FACE_MATCHING_SERVER = 'http://localhost:5001';

  // Capture current frame as base64 image
  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.9);
  };

  // Run the Wood Wide AI pipeline (train -> wait -> generate embeddings)
  const runPipeline = async () => {
    setIsProcessing(true);
    try {
      const response = await fetch(`${FACE_MATCHING_SERVER}/full-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timeout: 300 })
      });
      const data = await response.json();
      if (data.success) {
        setModelStatus('COMPLETE');
        alert(`Pipeline complete! ${data.embeddings_generated} embeddings created.`);
      } else {
        alert(`Pipeline failed: ${data.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
    setIsProcessing(false);
  };

  // Check model training status
  const checkStatus = async () => {
    try {
      const response = await fetch(`${FACE_MATCHING_SERVER}/model-status`);
      const data = await response.json();
      setModelStatus(data.status || 'NOT_TRAINED');
    } catch (error) {
      console.error('Status check failed:', error);
    }
  };

  // Match captured face against database using Wood Wide AI embeddings
  const matchFace = async (features) => {
    try {
      const response = await fetch(`${FACE_MATCHING_SERVER}/match-with-woodwide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, threshold: 0.85 })
      });
      const data = await response.json();
      setMatchResult(data);
      return data;
    } catch (error) {
      console.error('Match failed:', error);
      return null;
    }
  };

  // Extract features from image (calls your feature extraction server)
  const extractFeatures = async (imageData) => {
    try {
      const response = await fetch('http://localhost:5000/extract-features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });
      const data = await response.json();
      return data.success ? data.features : null;
    } catch (error) {
      console.error('Feature extraction failed:', error);
      return null;
    }
  };

  // Full capture and identify flow
  const captureAndIdentify = async () => {
    setIsProcessing(true);
    setMatchResult(null);

    // Step 1: Capture frame
    const imageData = captureFrame();
    if (!imageData) {
      alert('Failed to capture frame');
      setIsProcessing(false);
      return;
    }

    // Step 2: Extract features
    const features = await extractFeatures(imageData);
    if (!features) {
      alert('Failed to extract features');
      setIsProcessing(false);
      return;
    }

    // Step 3: Match using Wood Wide AI embeddings + Euclidean distance
    const match = await matchFace(features);
    
    if (match?.matched) {
      alert(`Identified: ${match.person_name} (${(match.confidence * 100).toFixed(1)}% confidence)`);
    } else {
      alert('No match found in database');
    }

    setIsProcessing(false);
  };

  useEffect(() => {
    const vision = new RealtimeVision({
      apiUrl: 'https://cluster1.overshoot.ai/api/v0.2',
      apiKey: 'ovs_3ca60448b9246224e080edb3159132a7',
      prompt: 'Read any visiso ble text',
      source: { type: "camera", cameraFacing: "environment" },
      onResult: (result) => {
        setResult(result.result);
      }
    });

    visionRef.current = vision;
    
    vision.start().then(() => {
      if (videoRef.current) {
        videoRef.current.srcObject = vision.getMediaStream();
      }
    });

    // Check model status on mount
    checkStatus();

    return () => vision.stop();
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline style={{width: '100%'}} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      {/* Control buttons */}
      <div style={{ padding: '10px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button onClick={runPipeline} disabled={isProcessing}>
          Train Wood Wide AI ({modelStatus})
        </button>
        <button onClick={checkStatus}>
          Check Status
        </button>
        <button onClick={captureAndIdentify} disabled={isProcessing || modelStatus !== 'COMPLETE'}>
          Capture & Identify
        </button>
      </div>

      {/* Match result display */}
      {matchResult && (
        <div style={{ 
          padding: '15px', 
          margin: '10px',
          backgroundColor: matchResult.matched ? '#e8f5e9' : '#fff3e0',
          borderRadius: '8px'
        }}>
          {matchResult.matched ? (
            <>
              <h3>Match Found!</h3>
              <p><strong>Name:</strong> {matchResult.person_name}</p>
              <p><strong>Confidence:</strong> {(matchResult.confidence * 100).toFixed(1)}%</p>
              <p><strong>Distance:</strong> {matchResult.distance?.toFixed(4)}</p>
            </>
          ) : (
            <h3>No Match Found</h3>
          )}
        </div>
      )}

      {/* Original result display */}
      <div>{result}</div>
    </div>
  );
}

export default VisionApp;