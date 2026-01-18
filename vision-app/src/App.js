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

  const captureScreenshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas) return null;
    
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    addLog(' Screenshot captured!');
    
    return imageData;
  };

  useEffect(() => {
    addLog(' Initializing RealtimeVision...');
    
    const vision = new RealtimeVision({
      apiUrl: 'https://cluster1.overshoot.ai/api/v0.2',
      apiKey: 'ovs_3ca60448b9246224e080edb3159132a7',
      
      //  START WITH SIMPLE PROMPT
      prompt: SIMPLE_PROMPT,
      
      // Simple schema for testing
      outputSchema: {
        type: "object",
        properties: {
          personFound: { type: "boolean" },
          description: { type: "string" }
        }
      },
      
      source: { type: "camera", cameraFacing: "environment" },
      
      processing: {
        fps: 30,
        sampling_ratio: 0.3, // Increased to 30% for more frequent checks
        clip_length_seconds: 1.0,
        delay_seconds: 1.0
      },
      
      debug: true, // Enable SDK debug logging
      
      onResult: (result) => {
        addLog(' onResult callback fired!');
        addLog(`Raw result: ${JSON.stringify(result).substring(0, 100)}...`);
        
        try {
          const data = JSON.parse(result.result);
          addLog(` Parsed: ${JSON.stringify(data)}`);
          setResult(JSON.stringify(data, null, 2));
          setStatus(`✓ Result received - Person: ${data.personFound}`);
          
          // Simple detection logic for testing
          if (data.personFound) {
            consecutiveDetections.current++;
            const count = consecutiveDetections.current;
            setStatus(`✓ Person detected ${count}/3 times`);
            addLog(`✓ Detection ${count}/3`);
            
            if (count >= 3) {
              addLog(' PERSON OF INTEREST CONFIRMED!');
              setStatus(' CONFIRMED! Capturing...');
              captureScreenshot();
              consecutiveDetections.current = 0;
            }
          } else {
            consecutiveDetections.current = 0;
            setStatus('Scanning... no person detected');
          }
        } catch (e) {
          addLog(` Parse error: ${e.message}`);
          addLog(`Raw result that failed: ${result.result}`);
          console.error('Full error:', e);
        }
      },
      
      onError: (error) => {
        addLog(` onError: ${error.message}`);
        console.error('Full error object:', error);
        setStatus('Error: ' + error.message);
      }
    });

    visionRef.current = vision;
    
    addLog(' Calling vision.start()...');
    
    vision.start()
      .then(() => {
        addLog(' vision.start() succeeded');
        setStatus('🎥 Camera active, waiting for results...');
        
        if (videoRef.current) {
          const stream = vision.getMediaStream();
          addLog(`📹 Got MediaStream: ${stream ? 'YES' : 'NO'}`);
          videoRef.current.srcObject = stream;
        }
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
        {result || 'No results yet'}
      </div>
    </div>
  );
}

export default VisionApp;