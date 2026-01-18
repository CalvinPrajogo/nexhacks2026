import { useEffect, useRef, useState } from 'react';
import { RealtimeVision } from '@overshoot/sdk';
import './App.css';

function App() {
  // State management
  const [appState, setAppState] = useState('initial'); // 'initial' | 'scanning' | 'results'
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [personInfo, setPersonInfo] = useState(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [completedSections, setCompletedSections] = useState(0);
  const [nameComplete, setNameComplete] = useState(false);
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const visionRef = useRef(null);
  const consecutiveDetections = useRef(0);

  // PERSON_OF_INTEREST_PROMPT
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
  "personDetected": true/false,
  "isCentered": true/false,
  "isStationary": true/false,
  "isEngaged": true/false,
  "visibilityQuality": "good/fair/poor",
  "personOfInterestFound": true/false,
  "details": {"reasoning": "explain decision"},
  "confidence": 0.0-1.0
}`;

  // Format name helper
  const formatName = (name) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  // Demo data
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

  // Capture screenshot
  const captureScreenshot = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);

    if (visionRef.current) {
      visionRef.current.stop();
    }

    await extractFaceFeatures(dataUrl);
  };

  // Extract face features
  const extractFaceFeatures = async (imageData: string) => {
    try {
      const response = await fetch('http://localhost:5000/extract-features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });

      const result = await response.json();

      if (result.success && result.features) {
        await matchFace(result.features);
      }
    } catch (error) {
      console.error('Feature extraction error:', error);
    }
  };

  // Match face
  const matchFace = async (features: any) => {
    try {
      const response = await fetch('http://localhost:5002/match-face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });

      const result = await response.json();

      if (result.success && result.matched) {
        const formattedName = formatName(result.person_name);
        setConfidence(result.confidence * 100);
        await researchPerson(formattedName);
      }
    } catch (error) {
      console.error('Matching error:', error);
    }
  };

  // Research person
  const researchPerson = async (personName: string) => {
    setIsResearching(true);

    // Check for demo data first
    if (demoData[personName]) {
      setTimeout(() => {
        setPersonInfo(demoData[personName]);
        setIsResearching(false);
      }, 1500);
      return;
    }

    // Fallback to API call (if needed)
    setIsResearching(false);
  };

  // Handle detection result
  const onResult = (data: any) => {
    if (!data?.result?.personOfInterestFound) {
      consecutiveDetections.current = 0;
      return;
    }

    consecutiveDetections.current++;

    if (consecutiveDetections.current >= 3) {
      consecutiveDetections.current = 0;
      setScanProgress(0);
      
      // Start scan animation
      const interval = setInterval(() => {
        setScanProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            captureScreenshot();
            return 100;
          }
          return prev + 2;
        });
      }, 80);
    }
  };

  // Initialize vision on start
  const handleStartClick = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setAppState('scanning');
      setIsTransitioning(false);
      initializeVision();
    }, 800);
  };

  // Initialize RealtimeVision
  const initializeVision = () => {
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
      pollingInterval: 1000,
      debug: true,
      onResult
    });

    visionRef.current = vision;

    vision.start()
      .then(() => {
        if (videoRef.current) {
          const stream = vision.getMediaStream();
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => {
        console.error('Vision start error:', err);
      });
  };

  // Watch for scan completion to transition to results
  useEffect(() => {
    if (personInfo && !isResearching && scanProgress === 100) {
      setTimeout(() => {
        setIsTransitioning(true);
        setTimeout(() => {
          setAppState('results');
          setIsTransitioning(false);
        }, 1000);
      }, 500);
    }
  }, [personInfo, isResearching, scanProgress]);

  // Reset function
  const handleReset = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setAppState('initial');
      setPersonInfo(null);
      setScanProgress(0);
      setConfidence(0);
      consecutiveDetections.current = 0;
      setIsTransitioning(false);
    }, 500);
  };

  return (
    <div style={{
      minHeight: '100vh',
      position: 'relative',
      overflow: 'hidden',
      background: 'linear-gradient(135deg, #020617, #1e3a8a, #020617)'
    }}>
      {/* Particle Background */}
      <div style={{ position: 'absolute', inset: 0 }}>
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1), transparent 50%)'
        }} />
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: 'linear-gradient(to right, #0f172a 1px, transparent 1px), linear-gradient(to bottom, #0f172a 1px, transparent 1px)',
          backgroundSize: '4rem 4rem',
          maskImage: 'radial-gradient(ellipse 80% 50% at 50% 50%, #000 70%, transparent 110%)',
          WebkitMaskImage: 'radial-gradient(ellipse 80% 50% at 50% 50%, #000 70%, transparent 110%)',
          opacity: 0.2
        }} />
      </div>

      {/* Main Content */}
      <div style={{
        position: 'relative',
        zIndex: 10,
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '3rem 1.5rem'
      }}>
        {/* Header */}
        <div style={{
          position: 'absolute',
          top: '2rem',
          left: '50%',
          textAlign: 'center',
          transition: 'all 0.7s',
          opacity: appState === 'initial' ? 1 : 0,
          transform: appState === 'initial' ? 'translate(-50%, 0)' : 'translate(-50%, -1rem)'
        }}>
          <h2 style={{
            fontSize: '0.875rem',
            fontWeight: 500,
            letterSpacing: '0.3em',
            color: 'rgba(96, 165, 250, 0.6)',
            textTransform: 'uppercase'
          }}>
            Neural Recognition System
          </h2>
        </div>

        {/* Initial State */}
        {appState === 'initial' && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '2rem',
            transition: 'all 0.7s',
            opacity: isTransitioning ? 0 : 1,
            transform: isTransitioning ? 'scale(0.95)' : 'scale(1)'
          }}>
            <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
              <h1 style={{
                fontSize: '2.5rem',
                fontWeight: 300,
                color: 'white',
                marginBottom: '1rem',
                letterSpacing: '-0.02em'
              }}>
                Person Detection
              </h1>
              <p style={{
                color: '#94a3b8',
                fontSize: '1.125rem',
                maxWidth: '28rem',
                margin: '0 auto'
              }}>
                Advanced AI-powered identity recognition using neural networks
              </p>
            </div>

            <button
              onClick={handleStartClick}
              style={{
                position: 'relative',
                padding: '1rem 2rem',
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(24px)',
                WebkitBackdropFilter: 'blur(24px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '9999px',
                transition: 'all 0.3s',
                cursor: 'pointer',
                color: 'white'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <svg
                  style={{ width: '1.25rem', height: '1.25rem' }}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
                Start Detection
              </span>
            </button>

            <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '1rem' }}>
              Click to initialize camera feed
            </p>
          </div>
        )}

        {/* Scanning State */}
        {appState === 'scanning' && (
          <div style={{
            width: '100%',
            maxWidth: '42rem',
            transition: 'all 1s',
            opacity: isTransitioning ? 0 : 1,
            transform: isTransitioning ? 'scale(0.95)' : 'scale(1)'
          }}>
            <div style={{
              position: 'relative',
              borderRadius: '0.75rem',
              overflow: 'hidden',
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              padding: '0.25rem'
            }}>
              <div style={{
                position: 'relative',
                borderRadius: '0.5rem',
                overflow: 'hidden',
                aspectRatio: '16 / 9',
                background: 'rgba(15, 23, 42, 0.5)'
              }}>
                {/* Video Feed */}
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline
                  muted
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />

                {/* Scanning overlay */}
                {scanProgress > 0 && (
                  <>
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      right: 0,
                      height: '0.125rem',
                      background: 'linear-gradient(to right, transparent, #60a5fa, transparent)',
                      top: `${scanProgress}%`,
                      opacity: scanProgress < 100 ? 1 : 0,
                      transition: 'opacity 0.3s ease'
                    }} />
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <div style={{
                        width: '8rem',
                        height: '8rem',
                        borderRadius: '50%',
                        border: '4px solid rgba(255, 255, 255, 0.2)',
                        position: 'relative'
                      }}>
                        <div style={{
                          position: 'absolute',
                          inset: 0,
                          borderRadius: '50%',
                          border: '4px solid #60a5fa',
                          clipPath: `polygon(50% 50%, 50% 0%, ${50 + 50 * Math.sin((scanProgress / 100) * Math.PI * 2)}% ${50 - 50 * Math.cos((scanProgress / 100) * Math.PI * 2)}%, 50% 50%)`
                        }} />
                        <div style={{
                          position: 'absolute',
                          inset: 0,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          <div style={{
                            width: '0.75rem',
                            height: '0.75rem',
                            borderRadius: '50%',
                            background: '#60a5fa',
                            animation: 'pulse 2s infinite'
                          }} />
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {/* Corner brackets */}
                <div style={{
                  position: 'absolute',
                  top: '1rem',
                  left: '1rem',
                  width: '2rem',
                  height: '2rem',
                  borderLeft: '2px solid rgba(96, 165, 250, 0.5)',
                  borderTop: '2px solid rgba(96, 165, 250, 0.5)'
                }} />
                <div style={{
                  position: 'absolute',
                  top: '1rem',
                  right: '1rem',
                  width: '2rem',
                  height: '2rem',
                  borderRight: '2px solid rgba(96, 165, 250, 0.5)',
                  borderTop: '2px solid rgba(96, 165, 250, 0.5)'
                }} />
                <div style={{
                  position: 'absolute',
                  bottom: '1rem',
                  left: '1rem',
                  width: '2rem',
                  height: '2rem',
                  borderLeft: '2px solid rgba(96, 165, 250, 0.5)',
                  borderBottom: '2px solid rgba(96, 165, 250, 0.5)'
                }} />
                <div style={{
                  position: 'absolute',
                  bottom: '1rem',
                  right: '1rem',
                  width: '2rem',
                  height: '2rem',
                  borderRight: '2px solid rgba(96, 165, 250, 0.5)',
                  borderBottom: '2px solid rgba(96, 165, 250, 0.5)'
                }} />

                {/* Progress bar */}
                {scanProgress > 0 && (
                  <div style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    height: '0.25rem',
                    background: 'rgba(255, 255, 255, 0.1)'
                  }}>
                    <div style={{
                      height: '100%',
                      background: '#60a5fa',
                      transition: 'all 0.1s linear',
                      width: `${scanProgress}%`
                    }} />
                  </div>
                )}
              </div>
            </div>

            {/* Status text */}
            <div style={{ marginTop: '1rem', textAlign: 'center' }}>
              <p style={{ fontSize: '0.875rem', color: '#94a3b8', fontWeight: 500 }}>
                {scanProgress === 0 && "Waiting for person of interest..."}
                {(scanProgress > 0 && scanProgress < 100) && (
                  <>
                    Analyzing facial features...{' '}
                    <span style={{ color: 'white', fontWeight: 600 }}>{scanProgress}%</span>
                  </>
                )}
                {scanProgress === 100 && <span style={{ color: 'white' }}>Analysis complete</span>}
              </p>
            </div>
          </div>
        )}

        {/* Results State */}
        {appState === 'results' && personInfo && (
          <div style={{
            width: '100%',
            transition: 'all 0.7s',
            opacity: isTransitioning ? 0 : 1
          }}>
            <div style={{
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '1rem',
              padding: '2.5rem',
              width: '100%',
              maxWidth: '42rem',
              margin: '0 auto'
            }}>
              {/* Name Header */}
              <div style={{
                marginBottom: '2rem',
                paddingBottom: '1.5rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h1 style={{
                  fontSize: '2rem',
                  fontWeight: 700,
                  color: 'white'
                }}>
                  {personInfo.name}
                </h1>
                <p style={{ color: '#94a3b8', marginTop: '0.5rem' }}>
                  Identity verified • Confidence: {confidence.toFixed(1)}%
                </p>
              </div>

              {/* Result Sections */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {[
                  { icon: '🎓', label: 'Education', content: personInfo.education },
                  { icon: '💼', label: 'Occupation', content: personInfo.occupation },
                  { icon: '📍', label: 'Hometown', content: personInfo.hometown },
                  { icon: '👥', label: 'Connections', content: personInfo.friends },
                  { icon: '👨‍👩‍👧‍👦', label: 'Family', content: personInfo.family },
                  { icon: '⭐', label: 'Notable', content: personInfo.notable_info },
                ].map((section, index) => (
                  <div
                    key={section.label}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '1rem',
                      opacity: 0,
                      animation: `fadeIn 0.6s ease forwards ${index * 100}ms`
                    }}
                  >
                    <span style={{
                      fontSize: '1.5rem',
                      flexShrink: 0,
                      marginTop: '0.125rem'
                    }}>{section.icon}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <span style={{
                        fontSize: '0.875rem',
                        fontWeight: 600,
                        color: '#94a3b8',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                      }}>
                        {section.label}
                      </span>
                      <p style={{
                        color: 'white',
                        marginTop: '0.25rem',
                        lineHeight: 1.6
                      }}>
                        {section.content}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Footer */}
              <div style={{
                marginTop: '2rem',
                paddingTop: '1.5rem',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <p style={{
                  fontSize: '0.75rem',
                  color: '#64748b',
                  textAlign: 'center'
                }}>
                  Scan completed at {new Date().toLocaleTimeString()} • Data sourced from public records
                </p>
              </div>
            </div>

            {/* Reset Button */}
            <div style={{ marginTop: '2rem', textAlign: 'center' }}>
              <button
                onClick={handleReset}
                style={{
                  position: 'relative',
                  padding: '0.75rem 1.5rem',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(24px)',
                  WebkitBackdropFilter: 'blur(24px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '9999px',
                  transition: 'all 0.3s',
                  cursor: 'pointer',
                  color: 'white'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg
                    style={{ width: '1rem', height: '1rem' }}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
                    />
                  </svg>
                  Scan Another
                </span>
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{
          position: 'absolute',
          bottom: '1.5rem',
          left: '50%',
          transform: 'translateX(-50%)'
        }}>
          <p style={{
            fontSize: '0.75rem',
            color: '#475569',
            letterSpacing: '0.05em'
          }}>
            v2.4.1 • Secure Connection
          </p>
        </div>
      </div>

      {/* Hidden canvas for screenshots */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;
