import { useEffect, useRef, useState } from 'react';
import { RealtimeVision } from '@overshoot/sdk';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

// Brinly's hardcoded profile data
const BRINLY_PROFILE = {
  name: "Brinly Richards",
  image: "/brinly_richards.png",
  education: [
    {
      institution: "University of California, Santa Cruz (UCSC)",
      major: "Cognitive Science",
      emphasis: "Artificial Intelligence / Human–Computer Interaction",
      status: "Senior (expected graduation June 2026)"
    },
    {
      institution: "Ohlone College",
      details: "Coursework completed (semester system)"
    },
    {
      institution: "Stanford University",
      details: "Participant and later Student Staff Mentor in CNI-X / Tech-X programs (AI & mental health innovation)"
    }
  ],
  occupation: [
    {
      title: "Founder & Builder",
      organization: "Kaira — women's health / holistic wellness AI platform"
    },
    {
      title: "Emerging Scholars Fellow",
      organization: "Active Minds (2025 cohort) — mental health research & innovation"
    },
    {
      title: "Student Staff Mentor",
      organization: "Stanford CNI-X / Tech-X (AI, ML, health innovation programs)"
    },
    {
      title: "Campus Brain Ambassador",
      organization: "Simply Neuroscience"
    },
    {
      title: "Research & Innovation Presenter",
      organization: "Presenter at Active Minds National Mental Health Conference"
    },
    {
      title: "Product / HCI / Digital Health Researcher",
      organization: "Focus on human-centered AI, biopsychosocial health, and evidence-based digital health design"
    }
  ],
  expertise: [
    "Human-Centered AI (HCAI)",
    "Digital health & mental health technology",
    "Women's health & health autonomy",
    "Cognitive science & biopsychosocial models",
    "Product design & UX research",
    "AI ethics, trust, and human oversight",
    "Health innovation & early-stage venture design"
  ],
  projects: [
    {
      name: "Kaira",
      description: "Women's health AI platform emphasizing evidence, trust, and intentional design"
    },
    {
      name: "Second Brain",
      description: "AI cognitive-offloading assistant concept - Won 2nd place at Women's Vibe Coding Hackathon"
    },
    {
      name: "UCSC Ideathon (2025)",
      description: "Featured participant/winner associated with Kaira"
    }
  ],
  skills: [
    "UX & product research",
    "Figma prototyping",
    "AI-driven personalization concepts",
    "Research synthesis & scientific communication",
    "Mental health advocacy & program leadership",
    "Cross-disciplinary collaboration (tech + health + social impact)"
  ],
  friends: [
    "Jasmine Ahluwalia (jaz.ahluwalia)",
    "Trinity Abraham (trinityaliese)"
  ],
  family: [
    "Richard Rajkumar (Possibly father - Works at Oracle)"
  ],
  affiliations: [
    "Active Minds",
    "Simply Neuroscience",
    "Stanford CNI-X / Tech-X",
    "UCSC Startup & Innovation Ecosystem"
  ]
};

function VisionApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const visionRef = useRef(null);
  const consecutiveDetections = useRef(0);
  const [status, setStatus] = useState('');
  const [capturedImages, setCapturedImages] = useState([]);
  const [isStarted, setIsStarted] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [researchStage, setResearchStage] = useState('');
  const [videoVisible, setVideoVisible] = useState(true);
  const [isResearching, setIsResearching] = useState(false);
  const [matchedProfile, setMatchedProfile] = useState(null);
  const [matchedPersonName, setMatchedPersonName] = useState(null);

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

  const extractAndMatchFace = async (imageData) => {
    try {
      console.log('[App] Extracting facial features from screenshot...');
      
      // Call the face feature extraction endpoint
      const extractResponse = await fetch('http://localhost:5003/extract-features', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData
        })
      });

      if (!extractResponse.ok) {
        throw new Error(`Failed to extract features: ${extractResponse.status}`);
      }

      const extractData = await extractResponse.json();
      
      if (!extractData.success) {
        throw new Error(extractData.error || 'Failed to extract features');
      }

      const features = extractData.features;
      console.log('[App] Features extracted, attempting match...');

      // Call the face matching endpoint
      const matchResponse = await fetch('http://localhost:5003/match-face', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: features,
          threshold: 1.0
        })
      });

      if (!matchResponse.ok) {
        throw new Error(`Failed to match face: ${matchResponse.status}`);
      }

      const matchData = await matchResponse.json();

      if (!matchData.success) {
        throw new Error(matchData.error || 'Failed to match face');
      }

      let personName = 'Unknown';

      if (matchData.matched) {
      personName = matchData.person_name;
      console.log(`[App] ✓ Match found: ${personName}, confidence: ${matchData.confidence}`);
    } else {
      const topMatches = matchData.top_matches || [];
      if (topMatches.length > 0) {
        const bestMatch = topMatches[0];
        personName = bestMatch.name;
        console.log(`[App] ⚠ Best match: ${personName}, distance: ${bestMatch.distance}`);
      }
    }

      // Load profile BEFORE returning
      const profile = await loadPersonProfile(personName);
      
      // Set both in one batch to avoid race conditions
      setMatchedPersonName(personName);
      setMatchedProfile(profile);
      
      return { success: true, personName, profile };

    } catch (error) {
      console.error('[App] Face matching error:', error);
      setMatchedPersonName('Error');
      setMatchedProfile(null);
      return { success: false, personName: 'Error', profile: null };
    }
  };


  // Then update your flow:
  setTimeout(async () => {
    const result = await extractAndMatchFace(imageData);
    await startDeepResearch(result.personName);
  }, 500);

  const startDeepResearch = async (detectedName) => {
    setIsResearching(true);
    
    const stages = [
      'Extracting facial features',
      'Matching against database',
      `Match found: ${detectedName}`,  // Now this will work!
      'Initiating deep research protocol',
      'Scanning social media profiles',
      'Analyzing professional networks',
      'Aggregating public records',
      'Compiling comprehensive profile'
    ];

    for (let i = 0; i < stages.length; i++) {
      setResearchStage(stages[i]);
      await new Promise(resolve => setTimeout(resolve, 1200));
    }

    setIsResearching(false);
    setShowProfile(true);
  };


  const loadPersonProfile = async (personName) => {
    // This would ideally load from a backend, but for now use hardcoded profiles
    const profiles = {
      'eden_brunner': BRINLY_PROFILE, // Using Brinly's profile as example
      'brinly_richards': BRINLY_PROFILE,
      'calvin_prajogo': {
        name: 'Calvin Prajogo',
        image: '/calvin_prajogo.png',
        education: [
          {
            institution: 'University Name',
            major: 'Computer Science',
            status: 'Student'
          }
        ],
        occupation: [],
        expertise: ['Software Development', 'AI'],
        projects: [],
        skills: ['Programming', 'Problem Solving'],
        friends: [],
        family: [],
        affiliations: []
      }
    };

    return profiles[personName.toLowerCase().replace(/\s+/g, '_')] || profiles['eden_brunner'];
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

    const capture = {
      timestamp: new Date(),
      image: imageData,
      detectionData: detectionData,
      id: Date.now()
    };
    setCapturedImages(prev => [...prev, capture]);

    // Fade out video and start research with face matching
    setVideoVisible(false);
    setTimeout(async () => {
      await extractAndMatchFace(imageData);
      startDeepResearch();
    }, 500);

    return imageData;
  };

  const startDeepResearch = async () => {
    setIsResearching(true);
    // Build stages AFTER we know the matched person
  const stages = [
    'Extracting facial features',
    'Matching against database',
    matchSuccess ? `Match found: ${matchedPersonName}` : 'No match found - using best guess',
    'Initiating deep research protocol',
    'Scanning social media profiles',
    'Analyzing professional networks',
    'Aggregating public records',
    'Compiling comprehensive profile'
  ];

    for (let i = 0; i < stages.length; i++) {
      setResearchStage(stages[i]);
      await new Promise(resolve => setTimeout(resolve, 1200));
    }

    setIsResearching(false);
    setShowProfile(true);
  };

  const startVision = () => {
    setIsStarted(true);
    setStatus('Initializing camera...');

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
      onResult: (result) => {
        try {
          const data = JSON.parse(result.result);

          if (data.personOfInterestFound) {
            consecutiveDetections.current++;
            const count = consecutiveDetections.current;
            setStatus(`Person detected ${count}/3`);

            if (count >= 3) {
              setStatus('Person confirmed! Capturing...');
              const screenshot = captureScreenshot(data);
              if (screenshot) {
                visionRef.current.stop();
                setStatus('');
              }
              consecutiveDetections.current = 0;
            }
          } else {
            consecutiveDetections.current = 0;
            setStatus(`Scanning... ${data.reasoning || ''}`);
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      },
      onError: (error) => {
        console.error("Vision error:", error);
        setStatus("Error: " + error.message);
      },
    });

    visionRef.current = vision;

    vision.start()
      .then(() => {
        setStatus('Camera active - Scanning for people...');

        if (videoRef.current) {
          const stream = vision.getMediaStream();
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => {
        console.error('Start error:', err);
        setStatus('Failed to start camera: ' + err.message);
      });
  };

  useEffect(() => {
    return () => {
      if (visionRef.current) {
        visionRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="App">
      <NetworkBackground />

      <motion.div
        className="content"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <AnimatePresence>
          {!isResearching && (
            <>
              <motion.h1
                className="app-title"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                Nexum
              </motion.h1>

              <motion.p
                className="app-subtitle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                Intelligent Recognition System
              </motion.p>
            </>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {!isStarted ? (
            <motion.div
              className="start-container"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <motion.button
                className="get-started-btn"
                onClick={startVision}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
              >
                Get Started
              </motion.button>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem' }}
            >
              <AnimatePresence>
                {videoVisible && (
                  <motion.div
                    className="video-container"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.6 }}
                  >
                    <motion.div
                      className="video-wrapper"
                      whileHover={{ scale: 1.02, y: -4 }}
                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    >
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                      />
                    </motion.div>
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {status && !isResearching && !showProfile && (
                  <motion.div
                    className="status-card"
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -20, scale: 0.95 }}
                    transition={{ duration: 0.4 }}
                    whileHover={{ scale: 1.02 }}
                  >
                    <motion.p
                      className="status-text"
                      key={status}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      {status}
                    </motion.p>
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {showProfile && (
                  <ProfileDisplay profile={BRINLY_PROFILE} />
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Centered research loading overlay */}
      <AnimatePresence>
        {isResearching && (
          <motion.div
            className="research-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div
              className="research-content"
              key={researchStage}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.6 }}
            >
              <motion.p className="research-stage-text">
                {researchStage}
                <motion.span
                  className="ellipsis"
                  animate={{ opacity: [0, 1, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                >
                  ...
                </motion.span>
              </motion.p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ProfileDisplay({ profile }) {
  return (
    <motion.div
      className="profile-container"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 30 }}
      transition={{ duration: 0.6 }}
    >
      <motion.div
        className="profile-header"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <motion.img
          src={profile.image}
          alt="Profile"
          className="profile-image"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3, type: "spring" }}
        />
        <TypewriterText text={profile.name} className="profile-name" delay={0.4} speed={50} />
      </motion.div>

      <ProfileSection title="Education" items={profile.education} delay={0.6} />
      <ProfileSection title="Occupation & Roles" items={profile.occupation} delay={0.8} />
      <ProfileList title="Areas of Expertise" items={profile.expertise} delay={1.0} />
      <ProfileSection title="Notable Projects" items={profile.projects} delay={1.2} />
      <ProfileList title="Skills & Competencies" items={profile.skills} delay={1.4} />
      <ProfileList title="Friends" items={profile.friends} delay={1.6} />
      <ProfileList title="Family" items={profile.family} delay={1.8} />
      <ProfileList title="Professional Affiliations" items={profile.affiliations} delay={2.0} />
    </motion.div>
  );
}

function ProfileSection({ title, items, delay }) {
  return (
    <motion.div
      className="profile-section"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.5 }}
    >
      <TypewriterText text={title} className="section-title" delay={delay} speed={30} />
      <div className="section-content">
        {items.map((item, index) => (
          <motion.div
            key={index}
            className="section-item"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: delay + 0.1 + (index * 0.1), duration: 0.4 }}
          >
            {item.institution && (
              <>
                <TypewriterText
                  text={item.institution}
                  className="item-title"
                  delay={delay + 0.15 + (index * 0.1)}
                  speed={20}
                />
                {item.major && (
                  <TypewriterText
                    text={`Major: ${item.major}`}
                    className="item-detail"
                    delay={delay + 0.2 + (index * 0.1)}
                    speed={15}
                  />
                )}
                {item.emphasis && (
                  <TypewriterText
                    text={`Emphasis: ${item.emphasis}`}
                    className="item-detail"
                    delay={delay + 0.25 + (index * 0.1)}
                    speed={15}
                  />
                )}
                {item.status && (
                  <TypewriterText
                    text={`Status: ${item.status}`}
                    className="item-detail"
                    delay={delay + 0.3 + (index * 0.1)}
                    speed={15}
                  />
                )}
                {item.details && (
                  <TypewriterText
                    text={item.details}
                    className="item-detail"
                    delay={delay + 0.2 + (index * 0.1)}
                    speed={15}
                  />
                )}
              </>
            )}
            {item.title && (
              <>
                <TypewriterText
                  text={item.title}
                  className="item-title"
                  delay={delay + 0.15 + (index * 0.1)}
                  speed={20}
                />
                <TypewriterText
                  text={item.organization}
                  className="item-detail"
                  delay={delay + 0.2 + (index * 0.1)}
                  speed={15}
                />
              </>
            )}
            {item.name && (
              <>
                <TypewriterText
                  text={item.name}
                  className="item-title"
                  delay={delay + 0.15 + (index * 0.1)}
                  speed={20}
                />
                <TypewriterText
                  text={item.description}
                  className="item-detail"
                  delay={delay + 0.2 + (index * 0.1)}
                  speed={15}
                />
              </>
            )}
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function ProfileList({ title, items, delay }) {
  return (
    <motion.div
      className="profile-section"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.5 }}
    >
      <TypewriterText text={title} className="section-title" delay={delay} speed={30} />
      <div className="section-content">
        {items.map((item, index) => (
          <motion.div
            key={index}
            className="list-item"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: delay + 0.1 + (index * 0.05), duration: 0.3 }}
          >
            <TypewriterText
              text={`• ${item}`}
              className="item-text"
              delay={delay + 0.15 + (index * 0.05)}
              speed={15}
            />
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function TypewriterText({ text, className, delay, speed = 30 }) {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      let index = 0;
      const interval = setInterval(() => {
        if (index <= text.length) {
          setDisplayedText(text.slice(0, index));
          index++;
        } else {
          clearInterval(interval);
          setIsComplete(true);
        }
      }, speed);

      return () => clearInterval(interval);
    }, delay * 1000);

    return () => clearTimeout(timer);
  }, [text, delay, speed]);

  return (
    <span className={className}>
      {displayedText}
      {!isComplete && <motion.span
        className="cursor"
        animate={{ opacity: [0, 1, 0] }}
        transition={{ duration: 0.8, repeat: Infinity }}
      >
        |
      </motion.span>}
    </span>
  );
}

function NetworkBackground() {
  const canvasRef = useRef(null);
  const nodesRef = useRef([]);
  const mouseRef = useRef({ x: 0, y: 0 });
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Create nodes
    const nodeCount = 50;
    nodesRef.current = Array.from({ length: nodeCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      radius: Math.random() * 3 + 2
    }));

    const handleMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener('mousemove', handleMouseMove);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Update and draw nodes
      nodesRef.current.forEach((node) => {
        // Move nodes
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off edges
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Mouse interaction
        const dx = mouseRef.current.x - node.x;
        const dy = mouseRef.current.y - node.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 100) {
          node.x -= dx * 0.02;
          node.y -= dy * 0.02;
        }

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(80, 80, 80, 0.5)';
        ctx.fill();
      });

      // Draw connections
      nodesRef.current.forEach((node, i) => {
        nodesRef.current.slice(i + 1).forEach((otherNode) => {
          const dx = node.x - otherNode.x;
          const dy = node.y - otherNode.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(otherNode.x, otherNode.y);
            ctx.strokeStyle = `rgba(120, 120, 120, ${0.3 * (1 - dist / 150)})`;
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return <canvas ref={canvasRef} className="network-canvas" />;
}

export default VisionApp;
