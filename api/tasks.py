"""
Background task manager for video processing.
"""
import os
import time
import uuid
import logging
from typing import Dict, Any, List, Optional

# In-memory task store for tracking progress
task_store: Dict[str, Dict[str, Any]] = {}

# Configure logging
logger = logging.getLogger(__name__)

def create_task(file_path: str, num_videos: int, target_duration: int, add_captions: bool) -> str:
    """Create a new video processing task and return its ID."""
    task_id = str(uuid.uuid4())
    task_store[task_id] = {
        "task_id": task_id,
        "file_path": file_path,
        "num_videos": num_videos,
        "target_duration": target_duration,
        "add_captions": add_captions,
        "status": "pending",
        "message": "Task created, waiting to start",
        "progress": 0,
        "videos": [],
        "file_urls": [],
        "created_at": time.time(),
        "updated_at": time.time()
    }
    return task_id

def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status and details by ID."""
    if task_id in task_store:
        logger.info(f"Found task {task_id} with status: {task_store[task_id]['status']}")
        return task_store[task_id]
    else:
        logger.warning(f"Task not found: {task_id}")
        # For debugging, show all known task IDs
        if task_store:
            logger.info(f"Available tasks: {list(task_store.keys())}")
        return None

def update_task(task_id: str, **kwargs) -> bool:
    """Update task with new information."""
    if task_id not in task_store:
        return False

    # Update only provided fields
    task = task_store[task_id]
    for key, value in kwargs.items():
        if key in task:
            task[key] = value

    # Always update the timestamp
    task["updated_at"] = time.time()
    return True

def update_progress(task_id: str, progress: int, message: str = None) -> None:
    """Update task progress percentage and optional message."""
    if task_id in task_store:
        task = task_store[task_id]
        task["progress"] = progress
        if message:
            task["message"] = message
        task["updated_at"] = time.time()