import { useEffect, useRef, useState } from 'react';
import { RealtimeVision } from '@overshoot/sdk';
import OpenAI from 'openai';

function VisionApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const visionRef = useRef(null);
  const consecutiveDetections = useRef(0);
  const [result, setResult] = useState('');
  const [status, setStatus] = useState('Initializing...');
  const [debugLog, setDebugLog] = useState([]);
  const [capturedImages, setCapturedImages] = useState([]);
  const [personInfo, setPersonInfo] = useState(null);
  const [isResearching, setIsResearching] = useState(false);
  const [videoStopped, setVideoStopped] = useState(false);

  // OpenAI client - API key from environment variable
  const openai = new OpenAI({
    apiKey: process.env.REACT_APP_OPENAI_API_KEY,
    dangerouslyAllowBrowser: true // Only for development
  });

  // Add to debug log
  const addLog = (message) => {
    console.log(message);
    setDebugLog(prev => [...prev.slice(-10), `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  // Format name: calvin_prajogo -> Calvin Prajogo
  const formatName = (name) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
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
    addLog('📸 Screenshot captured!');

    const capture = {
      timestamp: new Date(),
      image: imageData,
      detectionData: detectionData,
      id: Date.now()
    };
    setCapturedImages(prev => [...prev, capture]);

    addLog('🚀 Starting feature extraction pipeline...');
    
    // Automatically extract facial features and match
    extractFaceFeatures(imageData);

    return imageData;
  };

  const extractFaceFeatures = async (imageData) => {
    try {
      addLog('🔍 Extracting facial features...');
      
      const response = await fetch('http://localhost:5001/extract-features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });
      
      const result = await response.json();
      
      if (result.success) {
        addLog(`✓ Features extracted! Total: ${result.feature_count}`);
        
        // Immediately match against database
        await matchFace(result.features);
      } else {
        addLog(`✗ Feature extraction failed: ${result.error}`);
      }
    } catch (error) {
      addLog(`✗ Server error: ${error.message}`);
      addLog('   Make sure feature server is running on port 5001');
    }
  };

  const researchPerson = async (personName) => {
    try {
      setIsResearching(true);
      addLog(`🔍 Researching ${personName}...`);

      // Hardcoded demo data
      const demoData = {
        "Brinly Richards": {
          name: "Brinly Richards",
          education: "Current: Ohlone College (California) - Cognitive Science (Undergraduate in progress). Previous: University of California, Santa Cruz - Cognitive Science. Transfer plans: Four-year university (Fall 2025), Major in Cognitive Science with intended minor in Health Innovation & Entrepreneurship.",
          occupation: "Emerging Scholars Fellow (2025) at Active Minds - mental health advocacy nonprofit. Fellowship focus: 'Mental Health: A Holistic Approach Accessible for All'. Areas of work: Mental health education and advocacy, psychology and neuroscience-adjacent research. LinkedIn: 'AI in Health Tech | Cognitive Science Major'",
          hometown: "Bay Area, California (specific hometown not publicly stated)",
          friends: "Appears in Instagram posts from student teams and academic event accounts. Likely connections include classmates, teammates, and academic collaborators. No publicly verifiable list of close friends available.",
          family: "No confirmed public information about parents, siblings, or other family members. Similar-name accounts exist but cannot be verified as related.",
          notable_info: "Instagram: @brinlyr (personal account with student life, group/team activities, event participation). LinkedIn profile focuses on cognitive science and AI in health tech. Active in mental health advocacy and academic research."
        }
      };

      // Check if we have hardcoded data
      if (demoData[personName]) {
        addLog('✓ Research completed (demo data)');
        setPersonInfo(demoData[personName]);
        setIsResearching(false);
        return;
      }

      const prompt = `Task: Conduct a comprehensive research profile on ${personName} to create a detailed bibliography and online presence map.

Objective: Demonstrate your ability to:
- Search across multiple platforms and data sources
- Compile and organize information systematically
- Provide accurate citations and source attribution
- Present findings in a structured, bibliographic format

Required Outputs:

Part 1: Platform Inventory
List all platforms where you found information about ${personName}, including but not limited to:
- Professional networks (LinkedIn, GitHub, etc.)
- Academic platforms (Google Scholar, ResearchGate, ORCID, etc.)
- Social media (Twitter/X, Instagram, Facebook, etc.)
- Personal websites, blogs, or portfolios
- Publication databases
- Conference proceedings
- News mentions or press coverage
- Any other relevant sources

For each platform, provide:
- Platform name
- Profile URL (if public)
- Date last updated (if visible)
- Type of content found

Part 2: Biographical Summary
Provide a comprehensive summary including:
- Professional background and current role
- Educational background
- Areas of expertise/specialization
- Notable projects or contributions
- Skills and competencies
- Professional affiliations

Part 3: Bibliography
Compile a complete bibliography of:
- Published papers, articles, or blog posts
- GitHub repositories or open-source contributions
- Conference presentations or talks
- Patents or technical documentation
- Any other published or publicly shared work

Part 4: Research Quality Assessment
- Confidence level for each piece of information (High/Medium/Low)
- Note any conflicting information across sources
- Identify gaps in available information
- Timestamp your research

Instructions:
- Only use publicly available information
- Cite all sources with URLs where applicable
- If you cannot find information, explicitly state this
- Organize findings in a clear, scannable format
- Note the date and time you conducted this research

CRITICAL: At the very end of your response, provide a JSON object with the following exact format containing SPECIFIC, REAL information you found (not generic placeholders):

{
  "name": "${personName}",
  "education": "Specific school/university and degree information found",
  "occupation": "Specific current job title and company name found",
  "hometown": "Specific city/location found",
  "friends": "Names of connections/colleagues found on social media or professional networks",
  "family": "Any family members mentioned publicly",
  "notable_info": "Specific achievements, publications, or other standout information"
}

Make sure the JSON contains ACTUAL specific information you discovered, not generic descriptions.`;

      const completion = await openai.chat.completions.create({
        model: 'gpt-5.2',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.3, // Lower temperature for more factual responses
      });

      const response = completion.choices[0].message.content;
      addLog('✓ Research completed');
      console.log('Full GPT Response:', response); // Debug log
      
      // Extract key info from response
      const keyInfo = extractKeyInfo(response, personName);
      setPersonInfo(keyInfo);
      
    } catch (error) {
      addLog(`✗ Research error: ${error.message}`);
      setPersonInfo({
        name: personName,
        education: 'Research failed - ' + error.message,
        occupation: 'Research failed',
        hometown: 'Research failed',
        friends: 'Research failed',
        family: 'Research failed',
        error: error.message
      });
    } finally {
      setIsResearching(false);
    }
  };

  const extractKeyInfo = (response, personName) => {
    // Try to extract JSON from the end of the response
    // Look for the last JSON object in the response
    const jsonMatches = response.match(/\{[\s\S]*?"education"[\s\S]*?\}/g);
    
    if (jsonMatches && jsonMatches.length > 0) {
      // Get the last JSON match (most likely to be the summary)
      const lastJson = jsonMatches[jsonMatches.length - 1];
      try {
        const parsed = JSON.parse(lastJson);
        console.log('Extracted JSON:', parsed); // Debug log
        return {
          name: personName,
          ...parsed
        };
      } catch (e) {
        console.error('JSON parse error:', e);
        // Continue to fallback
      }
    }

    // More aggressive fallback parsing - extract full sections
    const sections = {
      name: personName,
      education: extractSection(response, ['Educational Background:', 'Education:']),
      occupation: extractSection(response, ['Professional Background:', 'Current Role:', 'Occupation:']),
      hometown: extractSection(response, ['Location:', 'Hometown:', 'Based in:']),
      friends: extractSection(response, ['Connections:', 'Colleagues:', 'Friends:']),
      family: extractSection(response, ['Family:', 'Relatives:']),
      notable_info: extractSection(response, ['Notable', 'Achievements:', 'Expertise:'])
    };

    return sections;
  };

  const extractSection = (text, headers) => {
    for (const header of headers) {
      const regex = new RegExp(`${header}\\s*([\\s\\S]{20,500}?)(?=\\n\\n|\\n[A-Z][^:]*:|$)`, 'i');
      const match = text.match(regex);
      if (match && match[1].trim().length > 10) {
        return match[1].trim();
      }
    }
    return 'Information not found in research';
  };

  const extractField = (text, keywords) => {
    for (const keyword of keywords) {
      const regex = new RegExp(`${keyword}[:\\s-]*([^\\n]{10,200})`, 'i');
      const match = text.match(regex);
      if (match) {
        return match[1].trim();
      }
    }
    return 'Information not found';
  };

  const matchFace = async (features) => {
    try {
      addLog('🎯 Matching face against database...');
      
      const response = await fetch('http://localhost:5002/match-face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      
      const result = await response.json();
      
      if (result.success && result.matched) {
        const formattedName = formatName(result.person_name);
        addLog(`✓ MATCH FOUND: ${formattedName}`);
        addLog(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
        addLog(`   Distance: ${result.distance.toFixed(4)}`);
        setResult(`Identified: ${formattedName} (${(result.confidence * 100).toFixed(1)}% confidence)`);
        
        // Start research on the matched person with formatted name
        await researchPerson(formattedName);
      } else if (result.success && !result.matched) {
        addLog('✗ No match found in database');
        setResult('Unknown person');
      } else {
        addLog(`✗ Matching failed: ${result.error}`);
      }
    } catch (error) {
      addLog(`✗ Matching error: ${error.message}`);
      addLog('   Make sure to run: python3 vision-app/src/face_matching_server.py (port 5002)');
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

        pollingInterval: 1000, // Process frames every 1 second

        debug: true, // Enable SDK debug logging

        onResult: (result) => {
            addLog("✓ onResult callback fired!");
            
            try {
                const data = JSON.parse(result.result);
                console.log('Parsed result:', data);
                setResult(JSON.stringify(data, null, 2));
                
                // Check if person of interest found
                if (data.personOfInterestFound) {
                    consecutiveDetections.current++;
                    const count = consecutiveDetections.current;
                    setStatus(`Person of interest detected ${count}/3 times`);
                    addLog(`✓ Detection ${count}/3 - Confidence: ${data.confidence}`);
                    
                    // Require 3 consecutive detections to confirm
                    if (count >= 3) {
                        addLog('✓ PERSON OF INTEREST CONFIRMED! Capturing...');
                        setStatus('CONFIRMED! Capturing...');
                        const screenshot = captureScreenshot(data);
                        if (screenshot) {
                            addLog('✓ Screenshot captured - processing...');
                            addLog('✓ Stopping video feed...');
                            visionRef.current.stop();
                            setStatus('Processing complete - Camera stopped');
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
        addLog('✓ vision.start() succeeded');
        setStatus('Camera active, waiting for results...');
        
        if (videoRef.current) {
          const stream = vision.getMediaStream();
          addLog(`Got MediaStream: ${stream ? 'YES' : 'NO'}`);
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
        style={{ 
          width: '100%', 
          maxWidth: '640px', 
          border: '2px solid #ccc',
          display: videoStopped ? 'none' : 'block'
        }} 
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Person Research Results */}
      {personInfo && (
        <div style={{ 
          marginTop: '20px', 
          padding: '20px', 
          background: '#f9f9f9', 
          border: '2px solid #4CAF50',
          borderRadius: '8px',
          maxWidth: '640px'
        }}>
          <h2 style={{ color: '#4CAF50', marginTop: 0 }}>✓ Person Identified: {personInfo.name}</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '15px', marginTop: '15px' }}>
            <div>
              <strong style={{ color: '#333' }}>🎓 Education:</strong>
              <p style={{ marginTop: '5px', marginLeft: '20px' }}>{personInfo.education || 'Not available'}</p>
            </div>
            
            <div>
              <strong style={{ color: '#333' }}>💼 Occupation:</strong>
              <p style={{ marginTop: '5px', marginLeft: '20px' }}>{personInfo.occupation || 'Not available'}</p>
            </div>
            
            <div>
              <strong style={{ color: '#333' }}>📍 Hometown:</strong>
              <p style={{ marginTop: '5px', marginLeft: '20px' }}>{personInfo.hometown || 'Not available'}</p>
            </div>
            
            <div>
              <strong style={{ color: '#333' }}>👥 Friends/Connections:</strong>
              <p style={{ marginTop: '5px', marginLeft: '20px' }}>{personInfo.friends || 'Not available'}</p>
            </div>
            
            <div>
              <strong style={{ color: '#333' }}>👨‍👩‍👧‍👦 Family:</strong>
              <p style={{ marginTop: '5px', marginLeft: '20px' }}>{personInfo.family || 'Not available'}</p>
            </div>
            
            {personInfo.notable_info && (
              <div>
                <strong style={{ color: '#333' }}>⭐ Notable Information:</strong>
                <p style={{ marginTop: '5px', marginLeft: '20px' }}>{personInfo.notable_info}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {isResearching && (
        <div style={{ marginTop: '20px', padding: '15px', background: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px', maxWidth: '640px' }}>
          <strong>🔍 Researching person profile...</strong>
          <p style={{ marginTop: '5px' }}>This may take a moment as we gather information.</p>
        </div>
      )}
      
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