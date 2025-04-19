"""
Router for video processing endpoints.
"""
import os
import uuid
import shutil
import logging
import asyncio
from typing import Dict, List

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.schemas import VideoRequest, VideoResponse, VideoOutput, VideoStatus, VideoTask

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/videos", tags=["videos"])

# In-memory task storage (use a database in production)
tasks: Dict[str, VideoTask] = {}


@router.post("/upload", response_model=VideoResponse)
async def upload_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        num_videos: int = Form(1),
        target_duration: int = Form(60),
        add_captions: bool = Form(True),
):
    """
    Upload a video file for processing into multiple short videos with captions.
    """
    # Validate parameters
    if num_videos < 1 or num_videos > 5:
        raise HTTPException(status_code=400, detail="Number of videos must be between 1 and 5")

    # Create unique task ID and temporary file path
    task_id = str(uuid.uuid4())
    temp_file = f"temp_{task_id}_{file.filename}"

    # Save uploaded file
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded file")

    # Create task
    task = VideoTask(
        task_id=task_id,
        file_path=temp_file,
        num_videos=num_videos,
        target_duration=target_duration,
        add_captions=add_captions,
        status=VideoStatus.PENDING,
    )
    tasks[task_id] = task

    # Start background processing
    background_tasks.add_task(process_video, task_id)

    return VideoResponse(
        task_id=task_id,
        status=VideoStatus.PENDING,
        message=f"Processing started. Creating {num_videos} video(s) with{'out' if not add_captions else ''} captions.",
    )


@router.get("/status/{task_id}", response_model=VideoResponse)
async def get_status(task_id: str):
    """Get the status of a video processing task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    return VideoResponse(
        task_id=task.task_id,
        status=task.status,
        message=task.message,
    )


@router.get("/results/{task_id}", response_model=VideoOutput)
async def get_results(task_id: str):
    """Get the results of a completed video processing task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task.status != VideoStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task not completed. Current status: {task.status}")

    # Generate URLs for the video files
    base_url = f"/static/output/{task_id}"
    file_urls = [f"{base_url}/{video}" for video in task.videos]

    return VideoOutput(
        task_id=task.task_id,
        status=task.status,
        message=task.message,
        videos=task.videos,
        file_urls=file_urls,
    )


async def process_video(task_id: str):
    """Process the video in the background."""
    task = tasks[task_id]

    try:
        # Update status
        task.status = VideoStatus.PROCESSING
        task.message = "Processing video..."
        tasks[task_id] = task

        # Create output directory
        output_dir = f"static/output/{task_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Import necessary modules
        import sys
        from src import video_processing, audio_processing, speech_to_text
        from src import content_selection, video_editing, script_generation

        try:
            # Extract audio and process video
            video_info = video_processing.analyze_video(task.file_path)
            audio_file = audio_processing.extract_audio(task.file_path)

            # Transcribe audio
            transcript = speech_to_text.transcribe(audio_file, include_timestamps=True)

            # Clean transcript
            filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]
            clean_transcript = audio_processing.remove_filler_words(transcript, filler_words)

            # Select key moments
            selected_content = content_selection.select_key_moments(
                clean_transcript,
                target_duration=task.target_duration
            )

            # Generate script
            script = {
                'title': 'Short Video',
                'segments': []
            }

            # Check if selected_content is a list or a dict
            if isinstance(selected_content, list):
                script['segments'] = selected_content
            elif isinstance(selected_content, dict) and 'segments' in selected_content:
                script = selected_content
            else:
                # Create a simple script from the content
                for segment in clean_transcript:
                    if 'start' in segment and 'end' in segment and 'text' in segment:
                        script['segments'].append(segment)

            # Create output videos
            output_files = []

            if task.num_videos == 1:
                # Single video mode
                output_path = os.path.join(output_dir, "short_video.mp4")
                video_editing.create_short_video(
                    input_path=task.file_path,
                    output_path=output_path,
                    script=script,
                    add_captions=task.add_captions
                )
                output_files.append("short_video.mp4")
            else:
                # Multiple videos mode
                for i in range(task.num_videos):
                    output_files.append(f"short_video_{i + 1}.mp4")

                video_editing.create_multiple_short_videos(
                    input_path=task.file_path,
                    output_dir=output_dir,
                    script=script,
                    num_videos=task.num_videos,
                    target_duration=task.target_duration,
                    add_captions=task.add_captions
                )

            # IMPORTANT: Verify files were actually created
            files_exist = False
            existing_files = []
            for filename in output_files:
                full_path = os.path.join(output_dir, filename)
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    existing_files.append(filename)
                    files_exist = True

            # Update task status based on file verification
            if files_exist:
                task.videos = existing_files
                task.status = VideoStatus.COMPLETED
                task.message = f"Successfully created {len(existing_files)} short video(s)"
            else:
                task.status = VideoStatus.FAILED
                task.message = "Failed to create videos. Check server logs."

            tasks[task_id] = task

        except Exception as e:
            logger.error(f"Error in video processing pipeline: {e}")
            raise

        # Clean up temporary file
        if os.path.exists(task.file_path):
            os.remove(task.file_path)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        task.status = VideoStatus.FAILED
        task.message = f"Error: {str(e)}"
        tasks[task_id] = task

        # Clean up on error
        if os.path.exists(task.file_path):
            os.remove(task.file_path)


# async def create_multiple_short_videos(
#         input_path: str,
#         output_dir: str,
#         num_videos: int,
#         target_duration: int,
#         add_captions: bool
# ) -> List[str]:
#     """
#     Create multiple short videos with captions from a long video.
#     This function integrates with the existing Auto Video Summarizer code.
#     """
#     # This would be replaced with your actual code from src/
#     # For this example, I'll create a simple placeholder implementation
#
#     # Import the necessary modules from your existing code
#     import sys
#     import os
#
#     # Add the src directory to the path so we can import modules
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
#     from src import video_processing, audio_processing, speech_to_text, content_selection
#     from src import video_editing
#
#     # Process the video
#     try:
#         # 1. Extract audio
#         audio_path = audio_processing.extract_audio(input_path)
#
#         # 2. Transcribe audio
#         transcript = speech_to_text.transcribe(audio_path)
#
#         # 3. Clean transcript
#         filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]
#         clean_transcript = audio_processing.remove_filler_words(transcript, filler_words)
#
#         # 4. Divide content into logical segments for multiple videos
#         video_segments = divide_content_for_multiple_videos(clean_transcript, num_videos)
#
#         output_files = []
#
#         # 5. Process each segment into a separate video
#         for i, segment_transcript in enumerate(video_segments):
#             # Select key content within this segment
#             selected_content = content_selection.select_key_moments(
#                 segment_transcript,
#                 target_duration=target_duration // num_videos
#             )
#
#             # Create output file name
#             output_filename = f"short_video_{i + 1}.mp4"
#             output_path = os.path.join(output_dir, output_filename)
#
#             # Create short video
#             video_editing.create_short_video(
#                 input_path,
#                 output_path,
#                 selected_content,
#                 add_captions=add_captions
#             )
#
#             output_files.append(output_filename)
#
#         return output_files
#
#     except Exception as e:
#         logger.error(f"Error in video processing pipeline: {e}")
#         raise
#
#
# def divide_content_for_multiple_videos(transcript, num_videos):
#     """
#     Divide transcript into logical segments for multiple videos.
#     """
#     if not transcript or num_videos <= 1:
#         return [transcript]
#
#     # Get total duration
#     total_duration = 0
#     for segment in transcript:
#         if 'start' in segment and 'end' in segment:
#             total_duration += segment['end'] - segment['start']
#
#     # Simple time-based division for this example
#     # In a real implementation, you'd use NLP to find logical break points
#     segment_duration = total_duration / num_videos
#
#     result = [[] for _ in range(num_videos)]
#     current_duration = 0
#     current_segment = 0
#
#     for item in transcript:
#         if 'start' not in item or 'end' not in item:
#             # Add items without timing to the first segment
#             result[0].append(item)
#             continue
#
#         item_duration = item['end'] - item['start']
#
#         # If adding this item would exceed the current segment duration,
#         # move to the next segment (if we're not already at the last segment)
#         if current_duration + item_duration > segment_duration and current_segment < num_videos - 1:
#             current_segment += 1
#             current_duration = 0
#
#         result[current_segment].append(item)
#         current_duration += item_duration
#
#     # Filter out empty segments
#     return [segment for segment in result if segment]