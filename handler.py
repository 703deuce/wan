"""
RunPod Serverless Handler for Wan2.2-S2V-14B Model
Simple subprocess-based approach using generate.py
"""

import os
import json
import base64
import tempfile
import traceback
import subprocess
from typing import Dict, Any
import requests

# Constants
MODEL_DIR = "/app/models/Wan2.2-S2V-14B"
GENERATE_SCRIPT = "/app/wan2.2/generate.py"

# Global flag to track if model is downloaded
_model_downloaded = False


def ensure_model_downloaded():
    """Download model if it doesn't exist. Uses HUGGINGFACE_TOKEN from environment if needed."""
    global _model_downloaded
    
    if _model_downloaded:
        return
    
    # Check if model directory exists and has files
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        # Check for key model files
        required_files = ["config.json", "diffusion_pytorch_model.safetensors.index.json"]
        if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files):
            print(f"Model already exists at {MODEL_DIR}")
            _model_downloaded = True
            return
    
    # Download model using huggingface-cli
    print(f"Downloading model to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Get HF token from environment
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
    
    cmd = [
        "huggingface-cli", "download",
        "Wan-AI/Wan2.2-S2V-14B",
        "--local-dir", MODEL_DIR,
    ]
    
    if hf_token:
        cmd.extend(["--token", hf_token])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for model download
        )
        
        if result.returncode != 0:
            raise Exception(f"Model download failed: {result.stderr}")
        
        print("Model download completed successfully")
        _model_downloaded = True
        
    except subprocess.TimeoutExpired:
        raise Exception("Model download timed out after 1 hour")
    except Exception as e:
        raise Exception(f"Error downloading model: {str(e)}")


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


def run_generate_script(audio_path, image_path, output_path, prompt="", resolution="720", seed=None):
    """Run the Wan2.2 generate.py script for S2V generation."""
    # Calculate size based on resolution
    # Note: Temporarily using 768*512 for 480P to match current endpoint
    # Will update to 832*480 once new build completes
    if resolution == "480":
        size = "768*512"  # Temporary for old endpoint
    else:
        size = "1024*704"  # 720P as per docs
    
    # Build command following the exact format from Wan2.2 docs
    cmd = [
        "python", GENERATE_SCRIPT,
        "--task", "s2v-14B",
        "--size", size,
        "--ckpt_dir", MODEL_DIR,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--image", image_path,
        "--audio", audio_path,
    ]
    
    # Add output path if generate.py supports it (some versions do)
    if output_path:
        cmd.extend(["--output", output_path])
    
    if prompt:
        cmd.extend(["--prompt", prompt])
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run generate.py
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout
        cwd="/app/wan2.2"
    )
    
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise Exception(f"generate.py failed: {result.stderr}")
    
    print(f"Generation completed: {result.stdout}")
    
    # Check if output_path exists (if --output was used)
    if output_path and os.path.exists(output_path):
        return output_path
    
    # Fallback: Find the output video file (generate.py may output to current directory)
    # Get most recently modified video file in wan2.2 directory
    try:
        video_files = [
            os.path.join("/app/wan2.2", f) for f in os.listdir("/app/wan2.2")
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]
        if video_files:
            # Get the most recent one
            return max(video_files, key=os.path.getmtime)
    except:
        pass
    
    # Check output_path's directory as fallback
    if output_path:
        output_dir = os.path.dirname(output_path)
        try:
            video_files = [
                os.path.join(output_dir, f) for f in os.listdir(output_dir)
                if f.endswith(('.mp4', '.avi', '.mov'))
            ]
            if video_files:
                return max(video_files, key=os.path.getmtime)
        except:
            pass
    
    raise Exception("Could not find generated video file")


def process_request(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a video generation request.
    
    Expected input format:
    {
        "input": {
            "audio_url": "https://example.com/audio.wav",  # Required
            "image_url": "https://example.com/image.jpg",   # Required
            "prompt": "cinematic scene",  # Optional
            "resolution": "720",  # Optional: "480" or "720" (default: "720")
            "seed": 42,  # Optional: random seed
        }
    }
    """
    try:
        # Ensure model is downloaded before processing
        ensure_model_downloaded()
        
        input_data = job.get("input", {})
        
        # Validate required inputs
        audio_url = input_data.get("audio_url")
        if not audio_url:
            return {
                "error": "audio_url is required in input",
                "status": "error"
            }
        
        image_url = input_data.get("image_url")
        if not image_url:
            return {
                "error": "image_url is required. Wan2.2-S2V requires audio + image to generate video.",
                "status": "error"
            }
        
        # Get optional parameters
        prompt = input_data.get("prompt", "")
        resolution = input_data.get("resolution", "720")
        seed = input_data.get("seed", None)
        
        # Validate resolution
        if resolution not in ["480", "720"]:
            resolution = "720"
        
        # Create temporary directory for this job
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download audio file
            audio_path = os.path.join(tmpdir, "audio.wav")
            print(f"Downloading audio from {audio_url}...")
            download_file(audio_url, audio_path)
            
            # Download image file
            image_path = os.path.join(tmpdir, "image.jpg")
            print(f"Downloading image from {image_url}...")
            download_file(image_url, image_path)
            
            # Generate video using generate.py
            print(f"Generating video at {resolution}P resolution...")
            if prompt:
                print(f"Using text prompt: {prompt}")
            
            # Output video path
            video_path = os.path.join(tmpdir, "output.mp4")
            
            video_path = run_generate_script(
                audio_path=audio_path,
                image_path=image_path,
                output_path=video_path,
                prompt=prompt,
                resolution=resolution,
                seed=seed
            )
            
            # Read video file and encode to base64
            with open(video_path, 'rb') as f:
                video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            # Get video duration
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
            except:
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
    """
    try:
        job = event
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


# RunPod serverless handler
if __name__ == "__main__":
    try:
        import runpod
        print("Starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})
    except ImportError:
        print("RunPod SDK not available. Handler ready for testing.")
