"""
RunPod Serverless Handler for Wan2.2-S2V-14B Model
Handles audio-driven video generation requests
"""

import os
import sys
import json
import base64
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import torch
from PIL import Image
import numpy as np

# Add current directory and Wan2.2 repo to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/app/wan2.2")

# Try to import Wan2.2 modules - adjust based on actual structure
WanS2VPipeline = None
export_to_video = None

# Try multiple import paths
import_paths = [
    ("wan.pipelines", "WanS2VPipeline"),
    ("wan.utils", "export_to_video"),
]

for module_name, attr_name in import_paths:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        if attr_name == "WanS2VPipeline":
            WanS2VPipeline = getattr(module, attr_name, None)
        elif attr_name == "export_to_video":
            export_to_video = getattr(module, attr_name, None)
        print(f"Successfully imported {attr_name} from {module_name}")
    except ImportError as e:
        print(f"Failed to import {attr_name} from {module_name}: {e}")

# Try direct path imports
if not WanS2VPipeline:
    try:
        sys.path.insert(0, "/app/wan2.2/wan")
        from pipelines import WanS2VPipeline
        print("Successfully imported WanS2VPipeline from direct path")
    except ImportError as e:
        print(f"Direct path import failed: {e}")

if not export_to_video:
    try:
        sys.path.insert(0, "/app/wan2.2/wan")
        from utils import export_to_video
        print("Successfully imported export_to_video from direct path")
    except ImportError as e:
        print(f"Direct path import for export_to_video failed: {e}")

if not WanS2VPipeline:
    print("WARNING: WanS2VPipeline not found. Will need to use alternative loading method.")

# Global model instance
model = None
device = None


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


def download_model_if_needed():
    """Download the model using huggingface-cli if not already present."""
    model_dir = "/app/models/Wan2.2-S2V-14B"
    model_id = "Wan-AI/Wan2.2-S2V-14B"
    
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Model already exists at {model_dir}")
        return model_dir
    
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
    
    print(f"Downloading model {model_id} to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)
    
    # Use huggingface_hub to download
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            token=hf_token,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {model_dir}")
        return model_dir
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def initialize_model():
    """Initialize the Wan2.2-S2V model - model is loaded on-demand via generate.py approach."""
    global model, device
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download model if needed
    model_dir = download_model_if_needed()
    
    # The model will be loaded on-demand when processing requests
    # using the generate.py approach from Wan2.2
    print("Model directory ready. Model will be loaded on-demand during inference.")
    return model_dir


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
    global model, device
    
    try:
        # Initialize model if not already done
        if model is None:
            initialize_model()
        
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
            
            # Generate video
            print(f"Generating video at {resolution}P resolution...")
            if prompt:
                print(f"Using text prompt: {prompt}")
            
            # Adjust resolution parameters
            height = 480 if resolution == "480" else 720
            width = int(height * 16 / 9)  # 16:9 aspect ratio
            
            # Run inference with audio + image (required combination)
            with torch.no_grad():
                # Build model call parameters
                model_kwargs = {
                    "image": image,
                    "audio_path": audio_path,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
                
                # Add text prompt if provided
                if prompt:
                    model_kwargs["prompt"] = prompt
                
                # Add generator if seed provided
                if generator is not None:
                    model_kwargs["generator"] = generator
                
                # Generate video with audio + image conditioning
                output = model(**model_kwargs)
            
            # Extract video frames
            if hasattr(output, 'frames'):
                video_frames = output.frames[0] if isinstance(output.frames, list) else output.frames
            elif hasattr(output, 'images'):
                video_frames = output.images
            else:
                video_frames = output
            
            # Save video to temporary file
            video_path = os.path.join(tmpdir, "output.mp4")
            
            if export_to_video:
                export_to_video(video_frames, video_path, fps=24)
            else:
                # Fallback: use opencv or other method
                try:
                    import cv2
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, 24.0, (width, height))
                    for frame in video_frames:
                        if isinstance(frame, np.ndarray):
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                    out.release()
                except ImportError:
                    # Last resort: save as base64
                    pass
            
            # Read video file and encode to base64
            with open(video_path, 'rb') as f:
                video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            # Get video duration (approximate)
            duration = len(video_frames) / 24.0  # Assuming 24 fps
            
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

