import {RealTimeVision} from '@overshoot/sdk'

const vision = new RealtimeVision({
    apiUrl: 'https://cluster1.overshoot.ai/api/v0.2',
    apiKey: 'your_api_key_here',
    prompt: 'Read any visible text',
    source: { type: "camera", cameraFacing: "environment" }, // Add this for mobile camera
    onResult: (result) => {
        console.log(result.result);
        // Update the page too:
        document.getElementById('result').textContent = result.result;
    }
})

await vision.start()   // starts the camera and begins processing
await vision.stop()    // stops everything