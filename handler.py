"""
RunPod Serverless Handler for Wan2.2-S2V-14B Model
Handles audio-driven video generation requests using generate.py subprocess
"""

import os
import sys
import json
import base64
import tempfile
import traceback
import subprocess
from typing import Dict, Any
import requests


def download_file(url: str, output_path: str) -> str:
    """Download a file from URL to local path."""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download file from {url}: {str(e)}")


# Model directory (downloaded during Docker build)
MODEL_DIR = "/app/models/Wan2.2-S2V-14B"
GENERATE_SCRIPT = "/app/wan2.2/generate.py"


def run_generate_script(audio_path, image_path, output_path, prompt="", resolution="720", seed=None):
    """Run the Wan2.2 generate.py script for S2V generation."""
    import subprocess
    
    model_dir = "/app/models/Wan2.2-S2V-14B"
    generate_script = "/app/wan2.2/generate.py"
    
    # Calculate size based on resolution (maintaining aspect ratio from image)
    # For 720P, use 1024*704 as shown in examples
    # For 480P, use smaller size
    if resolution == "480":
        size = "768*512"  # Approximate 480P
    else:
        size = "1024*704"  # 720P
    
    # Build command
    cmd = [
        "python", generate_script,
        "--task", "s2v-14B",
        "--size", size,
        "--ckpt_dir", model_dir,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--image", image_path,
        "--audio", audio_path,
        "--output", output_path,
    ]
    
    if prompt:
        cmd.extend(["--prompt", prompt])
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd="/app/wan2.2"
        )
        
        if result.returncode != 0:
            raise Exception(f"generate.py failed: {result.stderr}")
        
        print(f"Generation completed. Output: {result.stdout}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Generation timed out after 30 minutes")
    except Exception as e:
        raise Exception(f"Error running generate.py: {str(e)}")

def process_request(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a video generation request.
    
    Expected input format:
    {
        "input": {
            "audio_url": "https://example.com/audio.wav",  # Required
            "image_url": "https://example.com/image.jpg",   # Required (visual condition)
            "prompt": "cinematic scene, professional lighting",  # Optional: text prompt for style
            "resolution": "720",  # Optional: "480" or "720" (default: "720")
            "seed": 42,  # Optional: random seed
            "num_inference_steps": 50,  # Optional: number of inference steps
            "guidance_scale": 1.0,  # Optional: guidance scale
        }
    }
    
    Note: Wan2.2-S2V requires audio + at least one visual condition (image).
    Audio alone is not supported - the model needs an image to animate.
    """
    try:
        input_data = job.get("input", {})
        
        # Validate required inputs
        audio_url = input_data.get("audio_url")
        if not audio_url:
            return {
                "error": "audio_url is required in input",
                "status": "error"
            }
        
        # Image is required - Wan2.2-S2V needs visual conditioning
        image_url = input_data.get("image_url")
        if not image_url:
            return {
                "error": "image_url is required. Wan2.2-S2V requires audio + image (visual condition) to generate video.",
                "status": "error"
            }
        
        # Get optional parameters
        prompt = input_data.get("prompt", "")  # Optional text prompt for style/camera/background
        resolution = input_data.get("resolution", "720")
        seed = input_data.get("seed", None)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        guidance_scale = input_data.get("guidance_scale", 1.0)
        
        # Validate resolution
        if resolution not in ["480", "720"]:
            resolution = "720"
        
        # Create temporary directory for this job
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download audio file
            audio_path = os.path.join(tmpdir, "audio.wav")
            print(f"Downloading audio from {audio_url}...")
            download_file(audio_url, audio_path)
            
            # Download image file (required)
            image_path = os.path.join(tmpdir, "image.jpg")
            print(f"Downloading image from {image_url}...")
            download_file(image_url, image_path)
            image = Image.open(image_path).convert("RGB")
            
            # Set up generator with seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
            
            # Generate video using Wan2.2 generate.py script
            print(f"Generating video at {resolution}P resolution...")
            if prompt:
                print(f"Using text prompt: {prompt}")
            
            # Output video path
            video_path = os.path.join(tmpdir, "output.mp4")
            
            # Run the generate.py script
            run_generate_script(
                audio_path=audio_path,
                image_path=image_path,
                output_path=video_path,
                prompt=prompt,
                resolution=resolution,
                seed=seed
            )
            
            # Check if video was generated
            if not os.path.exists(video_path):
                # Try to find the output file (generate.py might use different naming)
                possible_outputs = [
                    os.path.join(tmpdir, f) for f in os.listdir(tmpdir) 
                    if f.endswith(('.mp4', '.avi', '.mov'))
                ]
                if possible_outputs:
                    video_path = possible_outputs[0]
                else:
                    # Check wan2.2 directory
                    wan_outputs = [
                        os.path.join("/app/wan2.2", f) for f in os.listdir("/app/wan2.2") 
                        if f.endswith(('.mp4', '.avi', '.mov'))
                    ]
                    if wan_outputs:
                        video_path = wan_outputs[0]
                    else:
                        raise Exception("Video file was not generated")
            
            # Read video file and encode to base64
            with open(video_path, 'rb') as f:
                video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            # Get video duration using opencv or ffprobe
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
            except:
                # Fallback: estimate duration
                duration = 5.0  # Default estimate
            
            return {
                "status": "success",
                "video": video_base64,
                "resolution": resolution,
                "duration": duration,
                "format": "mp4",
                "fps": 24
            }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error processing request: {error_msg}")
        print(error_trace)
        
        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace
        }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler function.
    
    This function is called by RunPod for each job.
    """
    try:
        # RunPod passes the job in the event
        job = event
        
        # Process the request
        result = process_request(job)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Handler error: {error_msg}")
        print(error_trace)
        
        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace
        }


# Initialize model on module load (cold start optimization)
if __name__ == "__main__":
    # For RunPod serverless
    try:
        import runpod
        
        # Initialize model on startup for faster first request
        print("Initializing model on startup...")
        try:
            initialize_model()
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Warning: Model initialization failed: {str(e)}")
            print("Model will be initialized on first request.")
        
        # Start RunPod serverless worker
        runpod.serverless.start({"handler": handler})
        
    except ImportError:
        # For local testing without RunPod SDK
        print("RunPod SDK not available. Running in local test mode...")
        try:
            initialize_model()
            print("Model initialized. Ready to process requests.")
            
            # Example test request
            test_job = {
                "input": {
                    "audio_url": "https://example.com/test.wav",
                    "image_url": "https://example.com/image.jpg",
                    "prompt": "cinematic scene, professional lighting",
                    "resolution": "720"
                }
            }
            # Uncomment to test locally:
            # result = handler(test_job)
            # print(f"Test result: {result}")
            
        except Exception as e:
            print(f"Failed to initialize model: {str(e)}")
            sys.exit(1)
else:
    # When imported by RunPod, initialize on first request
    print("Handler module loaded. Model will initialize on first request.")

