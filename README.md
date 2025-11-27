# RunPod Serverless Endpoint for Wan2.2-S2V-14B

This repository contains a Docker-based RunPod serverless endpoint for the [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) audio-driven cinematic video generation model.

## Overview

This endpoint provides an API service that accepts audio (and optionally image) URLs, generates videos using the Wan2.2-S2V-14B model, and returns the generated video output. The service supports both 480P and 720P resolution generation.

## Features

- Audio-driven video generation with visual conditioning
- Required image input (model animates the image according to audio)
- Optional text prompt for style, camera, and background control
- Configurable resolution (480P or 720P)
- Base64-encoded video output
- Environment variable-based configuration
- GPU-optimized inference

## Prerequisites

- RunPod account with serverless access
- GitHub account (for repository deployment)
- Hugging Face account with access token

## Setup Instructions

### 1. Fork/Clone This Repository

Fork this repository to your GitHub account or clone it locally.

### 2. Configure Environment Variables

You'll need to set the following environment variables in RunPod:

- `HUGGINGFACE_TOKEN` - Your Hugging Face access token (required to download the model)
- `MODEL_CACHE_DIR` - Optional: Custom directory for model cache (default: `/app/models`)
- `TMP_DIR` - Optional: Custom directory for temporary files (default: `/app/tmp`)

### 3. Deploy to RunPod

1. **Connect GitHub to RunPod:**
   - Log in to [RunPod Console](https://console.runpod.io/)
   - Navigate to **Settings** → **Connections**
   - Click **Connect** next to GitHub
   - Authorize RunPod to access your repositories

2. **Create Serverless Endpoint:**
   - Go to **Serverless** section
   - Click **New Endpoint**
   - Select **Import Git Repository**
   - Choose your repository and branch
   - Configure the following:
     - **Dockerfile Path:** `Dockerfile` (root directory)
     - **Endpoint Name:** Choose a descriptive name
     - **Endpoint Type:** Queue or Load Balancer (based on your needs)
     - **GPU Configuration:** Select appropriate GPU (recommended: A100 or similar for 14B model)
     - **Environment Variables:** Add `HUGGINGFACE_TOKEN` and any other required variables

3. **Deploy:**
   - Click **Deploy Endpoint**
   - Wait for the build and deployment to complete

## API Usage

### Request Format

Send a POST request to your RunPod endpoint URL with the following JSON payload:

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "image_url": "https://example.com/image.jpg",
    "prompt": "cinematic scene, professional lighting",
    "resolution": "720",
    "seed": 42,
    "num_inference_steps": 50,
    "guidance_scale": 1.0
  }
}
```

### Parameters

- `audio_url` (required): URL to the audio file (WAV, MP3, etc.) - drives timing, lip-sync, and motion
- `image_url` (required): URL to a reference image (JPG, PNG, etc.) - visual condition that will be animated
- `prompt` (optional): Text prompt for style, camera work, and background (e.g., "cinematic scene, professional lighting")
- `resolution` (optional): Video resolution - `"480"` or `"720"` (default: `"720"`)
- `seed` (optional): Random seed for reproducibility
- `num_inference_steps` (optional): Number of denoising steps (default: 50)
- `guidance_scale` (optional): Guidance scale for generation (default: 1.0)

**Important:** Wan2.2-S2V requires both `audio_url` and `image_url`. The model animates the provided image according to the audio. Audio-only generation is not supported.

### Response Format

**Success Response:**
```json
{
  "status": "success",
  "video": "base64_encoded_video_data",
  "resolution": "720",
  "duration": 5.0,
  "format": "mp4",
  "fps": 24
}
```

**Error Response:**
```json
{
  "status": "error",
  "error": "Error message here",
  "traceback": "Detailed traceback (if available)"
}
```

### Example Usage

#### Using cURL

```bash
curl -X POST https://your-endpoint-id.runpod.net \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/speech.wav",
      "image_url": "https://example.com/image.jpg",
      "prompt": "cinematic scene, professional lighting",
      "resolution": "720"
    }
  }'
```

#### Using Python

```python
import requests
import base64

endpoint_url = "https://your-endpoint-id.runpod.net"

payload = {
    "input": {
        "audio_url": "https://example.com/speech.wav",
        "image_url": "https://example.com/reference.jpg",  # Required
        "prompt": "cinematic scene, professional lighting",  # Optional
        "resolution": "720",
        "seed": 42
    }
}

response = requests.post(endpoint_url, json=payload)
result = response.json()

if result["status"] == "success":
    # Decode base64 video
    video_data = base64.b64decode(result["video"])
    
    # Save video
    with open("output.mp4", "wb") as f:
        f.write(video_data)
    
    print(f"Video generated: {result['duration']}s at {result['resolution']}P")
else:
    print(f"Error: {result['error']}")
```

## Model Information

- **Model:** Wan2.2-S2V-14B
- **Repository:** [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
- **Paper:** [Wan-S2V: Audio-Driven Cinematic Video Generation](https://arxiv.org/abs/2508.18621)
- **License:** Apache 2.0

## Technical Details

- **Base Image:** NVIDIA CUDA 12.1.0 with cuDNN 8
- **Python Version:** 3.10
- **Framework:** PyTorch with CUDA support
- **Video Format:** MP4 (H.264)
- **Frame Rate:** 24 FPS
- **Supported Resolutions:** 480P (854x480) and 720P (1280x720)

## Troubleshooting

### Model Loading Issues

- Ensure `HUGGINGFACE_TOKEN` is set correctly
- Check that your Hugging Face account has access to the model
- Verify sufficient GPU memory (14B model requires significant VRAM)

### Download Failures

- Check that audio and image URLs are publicly accessible
- Verify file formats are supported (audio: WAV, MP3; image: JPG, PNG)
- Ensure URLs are valid and not expired
- Remember: both audio_url and image_url are required

### Memory Issues

- Use a GPU with sufficient VRAM (recommended: 40GB+ for 720P)
- Consider using 480P resolution for lower memory usage
- Reduce `num_inference_steps` if needed

## License

This project is licensed under the Apache 2.0 License, same as the Wan2.2 model.

## Acknowledgments

- [Wan-Video Team](https://github.com/Wan-Video) for the Wan2.2 model
- [RunPod](https://www.runpod.io/) for the serverless infrastructure
- [Hugging Face](https://huggingface.co/) for model hosting

## Support

For issues related to:
- **Model:** Check the [Wan2.2 GitHub repository](https://github.com/Wan-Video/Wan2.2)
- **RunPod:** Check [RunPod Documentation](https://docs.runpod.io/)
- **This Endpoint:** Open an issue in this repository

