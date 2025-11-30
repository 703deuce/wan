"""
Test RunPod Wan2.2-S2V endpoint with 480P resolution and 5-second audio
"""
import requests
import base64
import json
import time
import os
import sys
import subprocess

# RunPod API Configuration
RUNPOD_API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "vcj8g1qxre37mq"
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def download_file(url, output_path):
    """Download file from URL"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ Downloaded to {output_path}")

def trim_audio_ffmpeg(input_path, output_path, duration=5):
    """Trim audio to 5 seconds using ffmpeg"""
    print(f"Trimming audio to {duration} seconds using ffmpeg...")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-t", str(duration),
        "-y",  # Overwrite output file
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try with pydub as fallback
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            trimmed = audio[:duration * 1000]
            trimmed.export(output_path, format="wav")
            print(f"✓ Trimmed using pydub")
        except ImportError:
            print("Error: ffmpeg failed and pydub not available")
            print("STDERR:", result.stderr)
            raise Exception("Could not trim audio")
    else:
        print(f"✓ Trimmed using ffmpeg")

def upload_to_transfer_sh(file_path):
    """Upload file to transfer.sh"""
    print(f"Uploading {file_path} to transfer.sh...")
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            response = requests.put(
                f'https://transfer.sh/{filename}',
                data=f,
                headers={'Max-Downloads': '10', 'Max-Days': '7'},
                timeout=120
            )
            response.raise_for_status()
            url = response.text.strip()
            print(f"✓ Uploaded: {url}")
            return url
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nPlease upload the file manually and provide the URL")
        return None

def submit_job(audio_url, image_url, resolution="480"):
    """Submit a job to RunPod endpoint"""
    payload = {
        "input": {
            "audio_url": audio_url,
            "image_url": image_url,
            "resolution": resolution
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    print(f"Submitting job to RunPod (resolution: {resolution}P)...")
    response = requests.post(
        f"{RUNPOD_API_URL}/run",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    response.raise_for_status()
    return response.json()

def get_job_status(job_id):
    """Get the status of a RunPod job"""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    response = requests.get(
        f"{RUNPOD_API_URL}/status/{job_id}",
        headers=headers,
        timeout=30
    )
    
    response.raise_for_status()
    return response.json()

def main():
    print("=" * 60)
    print("Testing RunPod Wan2.2-S2V Endpoint - 480P, 5 seconds")
    print("=" * 60)
    
    # Audio and image URLs
    audio_url = "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/Anderson_News_Anchor.wav?alt=media&token=f989dd6f-c013-4e08-b1b0-c8316c1ce5a9"
    image_url = "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/13.jpg?alt=media&token=71fa3fa2-a853-4423-bcad-0d2931e8c00b"
    
    # Download and trim audio
    print("\nStep 1: Downloading and trimming audio to 5 seconds...")
    audio_original = "audio_original.wav"
    audio_trimmed = "audio_5s.wav"
    
    try:
        download_file(audio_url, audio_original)
        trim_audio_ffmpeg(audio_original, audio_trimmed, duration=5)
        
        # Upload trimmed audio
        print("\nStep 2: Uploading trimmed audio...")
        trimmed_audio_url = upload_to_transfer_sh(audio_trimmed)
        
        if not trimmed_audio_url:
            print("\n⚠ Could not upload trimmed audio. Using original audio URL.")
            print("   (Note: Original audio may be longer than 5 seconds)")
            trimmed_audio_url = audio_url
        
    except Exception as e:
        print(f"⚠ Warning: Could not trim audio: {e}")
        print("Using original audio URL")
        trimmed_audio_url = audio_url
    
    print(f"\nUsing:")
    print(f"  Audio: {trimmed_audio_url}")
    print(f"  Image: {image_url}")
    print(f"  Resolution: 480P")
    
    # Submit job
    print("\nStep 3: Submitting job to RunPod...")
    try:
        job_result = submit_job(trimmed_audio_url, image_url, resolution="480")
        job_id = job_result.get("id")
        
        if not job_id:
            print(f"❌ Error: No job ID returned.")
            print(f"Response: {json.dumps(job_result, indent=2)}")
            return
        
        print(f"✓ Job submitted: {job_id}")
        
    except Exception as e:
        print(f"❌ Error submitting job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response: {e.response.text}")
            except:
                pass
        return
    
    # Poll for results
    print("\nStep 4: Waiting for video generation...")
    print("(This may take several minutes - please wait)")
    
    max_wait_time = 1200  # 20 minutes
    check_interval = 5  # Check every 5 seconds
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            status = get_job_status(job_id)
            job_status = status.get("status")
            
            if job_status == "COMPLETED":
                output = status.get("output")
                if output and output.get("status") == "success":
                    # Download video
                    video_data = base64.b64decode(output["video"])
                    output_file = "output_video_480p_5s.mp4"
                    
                    with open(output_file, "wb") as f:
                        f.write(video_data)
                    
                    print(f"\n✅ SUCCESS!")
                    print(f"   Job ID: {job_id}")
                    print(f"   Video saved to: {output_file}")
                    print(f"   Resolution: {output.get('resolution')}P")
                    print(f"   Duration: {output.get('duration')} seconds")
                    print(f"\nYou can now play {output_file}")
                    return
                else:
                    error_msg = output.get("error", "Unknown error") if output else "No output"
                    print(f"\n❌ Job completed with error: {error_msg}")
                    if output and "traceback" in output:
                        print(f"\nTraceback:\n{output['traceback']}")
                    return
                    
            elif job_status == "FAILED":
                print(f"\n❌ Job failed")
                print(f"Details: {json.dumps(status, indent=2)}")
                return
                
            elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                print(f"  Status: {job_status}... (elapsed: {elapsed_time}s)")
                time.sleep(check_interval)
                elapsed_time += check_interval
            else:
                print(f"  Status: {job_status}...")
                time.sleep(check_interval)
                elapsed_time += check_interval
                
        except Exception as e:
            print(f"❌ Error checking job status: {e}")
            time.sleep(check_interval)
            elapsed_time += check_interval
    
    print(f"\n❌ Timeout: Job did not complete within {max_wait_time} seconds")
    print(f"Job ID: {job_id} - Check status manually in RunPod console")

if __name__ == "__main__":
    main()

