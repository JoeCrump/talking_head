"""
Storage module for file operations.
"""
import os
import logging
import requests
from typing import List, Optional, Dict

# Configure logging
logger = logging.getLogger(__name__)

# Constants
BUCKET_NAME = "videos"
USE_CLOUD_STORAGE = os.environ.get("USE_CLOUD_STORAGE", "false").lower() == "true"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def initialize_storage():
    """Initialize storage (create buckets if needed)."""
    if not USE_CLOUD_STORAGE or not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("Cloud storage not configured or disabled")
        return

    # Just log that we're using the existing bucket
    logger.info(f"Using existing Supabase bucket: '{BUCKET_NAME}'")


def upload_file(file_path: str, task_id: str) -> Optional[str]:
    """Upload a file to cloud storage."""
    if not USE_CLOUD_STORAGE or not SUPABASE_URL or not SUPABASE_KEY:
        # Return a local URL instead
        filename = os.path.basename(file_path)
        return f"/static/output/{task_id}/{filename}"

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        # Determine correct MIME type based on file extension
        mime_type = "video/mp4"  # Default for .mp4 files

        # Prepare headers with correct content type
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": mime_type
        }

        # Generate storage path
        filename = os.path.basename(file_path)
        storage_path = f"{task_id}/{filename}"

        logger.info(f"Uploading {file_path} to Supabase/{BUCKET_NAME}/{storage_path}")

        # Upload file
        with open(file_path, "rb") as f:
            file_data = f.read()

            upload_response = requests.post(
                f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{storage_path}",
                headers=headers,
                data=file_data
            )

        if upload_response.status_code in (200, 201):
            # Generate public URL
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{storage_path}"
            logger.info(f"Uploaded file to {public_url}")
            return public_url
        else:
            logger.warning(f"Failed to upload file: {upload_response.status_code} - {upload_response.text}")
            # Fall back to local URL
            return f"/static/output/{task_id}/{filename}"

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        # Fall back to local URL
        filename = os.path.basename(file_path)
        return f"/static/output/{task_id}/{filename}"

def upload_multiple(file_paths: List[str], task_id: str) -> List[str]:
    """Upload multiple files and return their URLs."""
    urls = []
    for file_path in file_paths:
        url = upload_file(file_path, task_id)
        if url:
            urls.append(url)
    return urls