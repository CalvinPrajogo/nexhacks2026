import { useEffect, useRef, useState } from 'react';
import { RealtimeVision } from '@overshoot/sdk';

function VisionApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const visionRef = useRef(null);
  const consecutiveDetections = useRef(0);
  const [result, setResult] = useState('');
  const [status, setStatus] = useState('Initializing...');
  const [debugLog, setDebugLog] = useState([]);
  const [capturedImages, setCapturedImages]  = useState([]);

  // Add to debug log
  const addLog = (message) => {
    console.log(message);
    setDebugLog(prev => [...prev.slice(-10), `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  // SIMPLIFIED PROMPT for testing
  const SIMPLE_PROMPT = `Detect if there is a person in the frame. Return JSON: {"personFound": true/false, "description": "brief description"}`;

  // Original detailed prompt (use this once simple one works)
  const PERSON_OF_INTEREST_PROMPT = `You are a person-of-interest detection system. Your goal is to identify when someone is intentionally positioning themselves in front of the camera.

DETECTION CRITERIA (ALL must be met):
1. PRESENCE: At least one person is clearly visible in the frame
2. POSITIONING: The person is in the CENTER THIRD of the frame (not at edges)
3. ENGAGEMENT: The person appears to be:
   - Facing the camera directly (not walking past)
   - Stationary or moving slowly (not rushing through)
   - Either talking, gesturing, or deliberately posing
4. VISIBILITY: Person's face and upper body are clearly visible (not blurry/occluded)

Return JSON format:
{
  "personDetected": boolean,
  "isCentered": boolean,
  "isStationary": boolean,
  "isEngaged": boolean,
  "visibilityQuality": "low" | "medium" | "high",
  "personOfInterestFound": boolean,
  "details": {
    "description": "detailed physical description",
    "position": "left/center/right",
    "activity": "what they're doing",
    "faceVisible": boolean,
    "bodyLanguage": "engaged/passing/distracted"
  },
  "confidence": number,
  "reasoning": "brief explanation of decision"
}

IMPORTANT: Set "personOfInterestFound" to TRUE only when someone is clearly centered, stationary, engaged, and visible.`;
  const downloadImage = (imageData, detectionData) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = `person-${new Date().getTime()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addLog(' Image downloaded to computer!');
  };

  const captureScreenshot = (detectionData) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas) return null;
    
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    addLog(' Screenshot captured!');

    const capture = {
      timestamp: new Date(),
      image: imageData,
      detectionData: detectionData,
      id: Date.now()
    };
    setCapturedImages(prev => [...prev, capture]);

    // Automatically extract facial features
    extractFaceFeatures(imageData);
    
    downloadImage(imageData, detectionData);

    return imageData;
  };

  const extractFaceFeatures = async (imageData) => {
    try {
      addLog('🔍 Extracting facial features...');
      
      const response = await fetch('http://localhost:5000/extract-features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });
      
      const result = await response.json();
      
      if (result.success) {
        addLog(`✅ Features extracted! Saved to: ${result.saved_to}`);
        addLog(`   Total features: ${result.feature_count}`);
      } else {
        addLog(`❌ Feature extraction failed: ${result.error}`);
      }
    } catch (error) {
      addLog(`⚠️ Server error: ${error.message}`);
      addLog('   Make sure to run: python3 feature_server.py');
    }
  };

  useEffect(() => {
    addLog(' Initializing RealtimeVision...');
    
    const vision = new RealtimeVision({
        apiUrl: "https://cluster1.overshoot.ai/api/v0.2",
        apiKey: "ovs_3ca60448b9246224e080edb3159132a7",

        prompt: PERSON_OF_INTEREST_PROMPT,

        outputSchema: {
            type: "object",
            properties: {
                personDetected: { type: "boolean" },
                isCentered: { type: "boolean" },
                isStationary: { type: "boolean" },
                isEngaged: { type: "boolean" },
                visibilityQuality: { type: "string" },
                personOfInterestFound: { type: "boolean" },
                details: { type: "object" },
                confidence: { type: "number" },
                reasoning: { type: "string" }
            },
            required: ["personOfInterestFound"]
        },

        source: { type: "camera", cameraFacing: "environment" },

        pollingInterval: 1000, // Process frames every 1 second (1000ms)

        debug: true, // Enable SDK debug logging

        onResult: (result) => {
            addLog("onResult callback fired!");
            
            try {
                const data = JSON.parse(result.result);
                console.log('Parsed result:', data);
                setResult(JSON.stringify(data, null, 2));
                
                // Check if person of interest is found
                if (data.personOfInterestFound) {
                    consecutiveDetections.current++;
                    const count = consecutiveDetections.current;
                    setStatus(`Person of interest detected ${count}/3 times`);
                    addLog(`Detection ${count}/3 - Confidence: ${data.confidence}`);
                    
                    // Require 3 consecutive detections to confirm
                    if (count >= 3) {
                        addLog('PERSON OF INTEREST CONFIRMED!');
                        setStatus('CONFIRMED! Capturing...');
                        const screenshot = captureScreenshot(data);
                        if (screenshot) {
                            addLog('Screenshot saved!');
                        }
                        consecutiveDetections.current = 0;
                    }
                } else {
                    // Reset counter if no person detected
                    if (consecutiveDetections.current > 0) {
                        addLog(`Detection chain broken (was at ${consecutiveDetections.current})`);
                    }
                    consecutiveDetections.current = 0;
                    setStatus(`Scanning... ${data.reasoning || 'No person of interest'}`);
                }
            } catch (e) {
                addLog(`Parse error: ${e.message}`);
                console.error('Parse error:', e, 'Raw:', result.result);
                setResult(result.result);
            }
        },

        onError: (error) => {
            addLog(` onError: ${error.message}`);
            console.error("Full error object:", error);
            setStatus("Error: " + error.message);
        },
    });

    visionRef.current = vision;
    
    addLog(' Calling vision.start()...');
    
    vision.start()
      .then(() => {
        addLog(' vision.start() succeeded');
        setStatus('Camera active, waiting for results...');
        
        // Log internal state
        console.log('Vision instance:', visionRef.current);
        
        if (videoRef.current) {
          const stream = vision.getMediaStream();
          addLog(`Got MediaStream: ${stream ? 'YES' : 'NO'}`);
          videoRef.current.srcObject = stream;
        }
        
        // Give it time to process
        setTimeout(() => {
          addLog('10 seconds elapsed - still waiting for results...');
        }, 10000);
      })
      .catch(err => {
        addLog(` vision.start() failed: ${err.message}`);
        console.error('Start error:', err);
        setStatus('Failed to start: ' + err.message);
      });

    return () => {
      addLog(' Cleanup: stopping vision');
      vision.stop();
    };
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>Person of Interest Detector</h1>
      
      <div style={{ marginBottom: '10px', fontSize: '18px', fontWeight: 'bold', color: status.includes('Error') ? 'red' : 'green' }}>
        Status: {status}
      </div>
      
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        style={{ width: '100%', maxWidth: '640px', border: '2px solid #ccc' }} 
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      <div style={{ marginTop: '20px' }}>
        <h3>Debug Log:</h3>
        <div style={{ background: '#000', color: '#0f0', padding: '10px', fontFamily: 'monospace', fontSize: '12px', maxHeight: '200px', overflow: 'auto' }}>
          {debugLog.map((log, i) => <div key={i}>{log}</div>)}
        </div>
      </div>
      
      <div style={{ marginTop: '20px', whiteSpace: 'pre-wrap', background: '#f0f0f0', padding: '10px' }}>
        <strong>Latest Result:</strong>
        {capturedImages.length > 0 && (
          <div style={{ marginTop: '20px' }}>
            <h3>📸 Captured Screenshots ({capturedImages.length})</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
              {capturedImages.map((capture) => (
                <div key={capture.id} style={{ border: '2px solid green', padding: '10px' }}>
                  <img src={capture.image} alt="Captured" style={{ width: '200px' }} />
                  <p style={{ fontSize: '12px' }}>{capture.timestamp.toLocaleTimeString()}</p>
                  <p style={{ fontSize: '11px' }}>{capture.detectionData?.details?.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default VisionApp;