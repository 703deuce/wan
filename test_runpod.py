"""
Test RunPod Wan2.2-S2V endpoint
Usage: python test_runpod.py [--audio-url URL] [--image-url URL]
"""
import requests
import base64
import json
import time
import os
import sys

# RunPod API Configuration
RUNPOD_API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "vcj8g1qxre37mq"
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

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
    
    print(f"Submitting job to RunPod...")
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
    print("Testing RunPod Wan2.2-S2V Endpoint")
    print("=" * 60)
    
    # Get URLs from command line or use defaults
    audio_url = None
    image_url = None
    
    if len(sys.argv) > 1:
        # Parse command line arguments
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--audio-url" and i + 1 < len(sys.argv):
                audio_url = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--image-url" and i + 1 < len(sys.argv):
                image_url = sys.argv[i + 1]
                i += 2
            else:
                i += 1
    
    # If no URLs provided, ask user
    if not audio_url:
        print("\nPlease provide audio file URL.")
        print("You can upload test.wav to https://file.io or https://transfer.sh")
        print("Or run: python test_runpod.py --audio-url <URL> --image-url <URL>")
        audio_url = input("Audio URL (or press Enter to exit): ").strip()
        if not audio_url:
            return
    
    if not image_url:
        print("\nPlease provide image file URL.")
        print("You can upload test.jpg to https://file.io or https://transfer.sh")
        image_url = input("Image URL (or press Enter to exit): ").strip()
        if not image_url:
            return
    
    print(f"\nUsing:")
    print(f"  Audio: {audio_url}")
    print(f"  Image: {image_url}")
    
    # Submit job
    print("\nStep 1: Submitting job to RunPod...")
    try:
        job_result = submit_job(audio_url, image_url)
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
    print("\nStep 2: Waiting for video generation...")
    print("(This may take several minutes - please wait)")
    
    max_wait_time = 600  # 10 minutes
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
                    output_file = "output_video.mp4"
                    
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

