"""
Quick script to check RunPod job status
"""
import requests
import json
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_job_status.py <job_id>")
        print("\nExample: python check_job_status.py 2504d183-8637-4a01-b115-ea62a84686dc-u1")
        sys.exit(1)
    
    job_id = sys.argv[1]
    print(f"Checking status for job: {job_id}\n")
    
    try:
        status = get_job_status(job_id)
        job_status = status.get("status")
        
        print(f"Status: {job_status}")
        print(f"Execution Time: {status.get('executionTime', 0) / 1000:.1f} seconds")
        
        if job_status == "COMPLETED":
            output = status.get("output")
            if output and output.get("status") == "success":
                print("\n✅ Job completed successfully!")
                print(f"   Resolution: {output.get('resolution')}P")
                print(f"   Duration: {output.get('duration')} seconds")
            else:
                print("\n❌ Job completed with error")
                print(f"Error: {output.get('error', 'Unknown') if output else 'No output'}")
        elif job_status == "FAILED":
            print("\n❌ Job failed")
            output = status.get("output", {})
            if output.get("error"):
                print(f"Error: {output['error']}")
        elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
            print(f"\n⏳ Job is {job_status.lower()}...")
            print("   This is normal - video generation can take 10-30+ minutes")
            print("   The process is running, please be patient.")
        
        print(f"\nFull status:")
        print(json.dumps(status, indent=2))
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")

