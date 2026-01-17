import {RealTimeVision} from 'overshoot/sdk'

const vision = new RealTimeVision({
    apiUrl: 'https://cluster1.overshoot.ai/api/v0.2',
    apiKey: 'your_api_key_here',
    prompt: 'Read any visible text',
    onResult: (result) => {
    console.log(result.result)
  }
})

await vision.start()   // starts the camera and begins processing
await vision.stop()    // stops everything