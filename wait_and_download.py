"""
Wait for a RunPod job to complete and download the video
"""
import requests
import base64
import json
import time
import sys

# RunPod API Configuration
RUNPOD_API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_ID = "vcj8g1qxre37mq"
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

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
    if len(sys.argv) < 2:
        print("Usage: python wait_and_download.py <job_id> [output_filename]")
        print("\nExample: python wait_and_download.py 2504d183-8637-4a01-b115-ea62a84686dc-u1")
        sys.exit(1)
    
    job_id = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"output_video_{job_id[:8]}.mp4"
    
    print("=" * 60)
    print(f"Waiting for job: {job_id}")
    print(f"Output will be saved to: {output_file}")
    print("=" * 60)
    print("\nWaiting for video generation...")
    print("(This may take 10-30+ minutes - please wait)\n")
    
    max_wait_time = 1800  # 30 minutes
    check_interval = 10  # Check every 10 seconds
    elapsed_time = 0
    last_status = None
    
    while elapsed_time < max_wait_time:
        try:
            status = get_job_status(job_id)
            job_status = status.get("status")
            
            # Only print status if it changed
            if job_status != last_status:
                print(f"[{elapsed_time//60}m {elapsed_time%60}s] Status: {job_status}")
                last_status = job_status
            
            if job_status == "COMPLETED":
                output = status.get("output")
                if output and output.get("status") == "success":
                    # Download video
                    print("\n✅ Job completed! Downloading video...")
                    video_data = base64.b64decode(output["video"])
                    
                    with open(output_file, "wb") as f:
                        f.write(video_data)
                    
                    print(f"\n✅ SUCCESS!")
                    print(f"   Job ID: {job_id}")
                    print(f"   Video saved to: {output_file}")
                    print(f"   Resolution: {output.get('resolution')}P")
                    print(f"   Duration: {output.get('duration')} seconds")
                    print(f"   Format: {output.get('format', 'mp4')}")
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
                output = status.get("output", {})
                if output.get("error"):
                    print(f"Error: {output['error']}")
                print(f"\nFull status:")
                print(json.dumps(status, indent=2))
                return
                
            elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                # Show progress every 30 seconds
                if elapsed_time % 30 == 0 and elapsed_time > 0:
                    exec_time = status.get("executionTime", 0) / 1000
                    print(f"  Still processing... (elapsed: {elapsed_time//60}m {elapsed_time%60}s, execution: {exec_time:.1f}s)")
                
                time.sleep(check_interval)
                elapsed_time += check_interval
            else:
                time.sleep(check_interval)
                elapsed_time += check_interval
                
        except Exception as e:
            print(f"❌ Error checking job status: {e}")
            time.sleep(check_interval)
            elapsed_time += check_interval
    
    print(f"\n❌ Timeout: Job did not complete within {max_wait_time} seconds")
    print(f"Job ID: {job_id} - Check status manually in RunPod console")
    print(f"You can check status with: python check_job_status.py {job_id}")

if __name__ == "__main__":
    main()

