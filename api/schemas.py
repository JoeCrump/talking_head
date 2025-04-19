"""
Pydantic models for request and response validation.
"""
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class VideoStatus(str, Enum):
    """Status of video processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoRequest(BaseModel):
    """Request model for video processing."""
    num_videos: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of short videos to create (max 5)"
    )
    target_duration: int = Field(
        default=60,
        ge=30,
        le=180,
        description="Target duration in seconds for each short video"
    )
    add_captions: bool = Field(
        default=True,
        description="Whether to add captions to the videos"
    )


class VideoResponse(BaseModel):
    """Response model for video processing status."""
    task_id: str
    status: VideoStatus
    message: str


class VideoOutput(BaseModel):
    """Response model for completed video processing."""
    task_id: str
    status: VideoStatus
    message: str
    videos: List[str]
    file_urls: List[str]


class VideoTask(BaseModel):
    """Internal model for video processing tasks."""
    task_id: str
    file_path: str
    num_videos: int
    target_duration: int
    add_captions: bool
    status: VideoStatus = VideoStatus.PENDING
    message: str = "Task created"
    videos: List[str] = []