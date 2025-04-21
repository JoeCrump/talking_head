"""
API routes for video processing.
"""
import os
import asyncio
import logging
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from api import tasks
from api import storage
from src import video_processing, audio_processing, speech_to_text, content_selection, video_editing

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


async def process_video_task(task_id: str, file_path: str, num_videos: int, target_duration: int, add_captions: bool):
    """Process video in background and update task status."""
    try:
        # Update status to processing
        tasks.update_task(task_id, status="processing", message="Starting video analysis", progress=5)

        # Step 1: Analyze video
        tasks.update_progress(task_id, 10, "Analyzing video")
        video_info = video_processing.analyze_video(file_path)

        # Step 2: Extract audio
        tasks.update_progress(task_id, 20, "Extracting audio")
        audio_file = audio_processing.extract_audio(file_path)

        # Step 3: Transcribe audio
        tasks.update_progress(task_id, 30, "Transcribing audio")
        transcript = speech_to_text.transcribe(audio_file)

        # Step 4: Select key moments
        tasks.update_progress(task_id, 50, "Selecting key moments")
        key_moments = content_selection.select_key_moments(transcript, target_duration)

        # Step 5: Create videos
        tasks.update_progress(task_id, 70, "Creating short videos")
        output_videos = video_editing.create_short_videos(
            file_path,
            key_moments["segments"],
            num_videos=num_videos,
            add_captions=add_captions,
            task_id=task_id
        )

        # Full paths to videos
        base_dir = os.path.join("static", "output", task_id)
        video_paths = [os.path.join(base_dir, video) for video in output_videos]

        # Step 6: Upload to cloud storage if configured
        tasks.update_progress(task_id, 85, "Uploading videos")
        file_urls = storage.upload_multiple(video_paths, task_id)

        # Update task with success
        tasks.update_task(
            task_id,
            status="completed",
            message=f"Successfully created {len(output_videos)} short video(s)",
            videos=output_videos,
            file_urls=file_urls,
            progress=100
        )

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        tasks.update_task(
            task_id,
            status="failed",
            message=f"Error: {str(e)}",
            progress=0
        )


@router.post("/upload")
async def upload_video(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        num_videos: int = Form(5),
        target_duration: int = Form(60),
        add_captions: bool = Form(True)
):
    """Upload a video for processing."""
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}_{video.filename}")

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())

        # Create a task
        task_id = tasks.create_task(file_path, num_videos, target_duration, add_captions)

        # Start processing in background
        background_tasks.add_task(
            process_video_task,
            task_id,
            file_path,
            num_videos,
            target_duration,
            add_captions
        )

        # Return task ID immediately
        return {"task_id": task_id, "status": "pending", "message": "Video processing started"}

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_video_status(task_id: str):
    """Get status of video processing task."""
    task = tasks.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Return different response fields based on status
    if task["status"] == "completed":
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "message": task["message"],
            "videos": task["videos"],
            "file_urls": task["file_urls"]
        }
    else:
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "message": task["message"],
            "progress": task["progress"]
        }


@router.get("/videos")
async def list_videos():
    """List all completed videos."""
    completed_tasks = [task for task in tasks.task_store.values() if task["status"] == "completed"]
    return completed_tasks