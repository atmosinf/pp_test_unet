import os
import argparse
from datetime import datetime
from huggingface_hub import HfApi
from dotenv import load_dotenv

def upload_model(local_path, repo_id, model_name=None, private=True):
    """
    Uploads a model file to Hugging Face Hub with a timestamped name.
    
    Format: repo_name_model_name_datetime.extension
    """
    # Load environment variables (for HF_TOKEN)
    load_dotenv()
    
    api = HfApi()
    
    # Extract details from local path if not provided
    if not model_name:
        model_name = os.path.basename(local_path).split('.')[0]
    
    extension = os.path.basename(local_path).split('.')[-1]
    repo_name_clean = repo_id.split('/')[-1]
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hf_filename = f"{repo_name_clean}_{model_name}_{timestamp}.{extension}"
    
    print(f"🚀 Preparing to upload {local_path} to {repo_id} as {hf_filename}...")
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        
        # Upload the file
        path_in_repo = hf_filename
        response = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"✅ Success! Model uploaded to: {response}")
        print(f"🔗 View at: https://huggingface.co/{repo_id}/blob/main/{path_in_repo}")
        
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        if "Unauthorized" in str(e):
            print("💡 Tip: Make sure you are logged in using 'huggingface-cli login' or have set HF_TOKEN in your .env file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub with timestamped naming.")
    parser.add_argument("local_path", help="Path to the local model file (e.g., unet.pth)")
    parser.add_argument("repo_id", help="Hugging Face repository ID (e.g., username/my-model)")
    parser.add_argument("--name", help="Custom model name (optional)", default=None)
    parser.add_argument("--public", action="store_false", dest="private", help="Make the repository public (default is private)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.local_path):
        print(f"❌ Error: Local file {args.local_path} not found.")
    else:
        upload_model(args.local_path, args.repo_id, args.name, args.private)
