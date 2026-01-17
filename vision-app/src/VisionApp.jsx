import { useEffect, useRef, useState } from 'react';
import { RealtimeVision } from '@overshoot/sdk';

function VisionApp() {
  const videoRef = useRef(null);
  const visionRef = useRef(null);
  const [result, setResult] = useState('');

  useEffect(() => {
    const vision = new RealtimeVision({
      apiUrl: 'https://cluster1.overshoot.ai/api/v0.2',
      apiKey: 'ovs_3ca60448b9246224e080edb3159132a7',
      prompt: 'Read any visible text',
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

    return () => vision.stop();
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline style={{width: '100%'}} />
      <div>{result}</div>
    </div>
  );
}

export default VisionApp;
