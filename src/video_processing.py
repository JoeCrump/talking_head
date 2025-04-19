"""
Video processing module for analyzing and segmenting videos.
"""

import logging
import os
import subprocess
import json
from typing import Dict, List, Tuple, Any

import ffmpeg
import numpy as np
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyze a video file and return metadata and scene information.
    
    Args:
        video_path: Path to the input video file.
        
    Returns:
        Dict containing video metadata and scene information.
    """
    logger.info(f"Analyzing video: {video_path}")
    
    try:
        # Get basic video metadata using ffprobe
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        # Extract basic properties
        width = int(video_info.get('width', 0))
        height = int(video_info.get('height', 0))
        fps = eval(video_info.get('r_frame_rate', '30/1'))
        duration = float(probe.get('format', {}).get('duration', 0))
        
        # Use MoviePy to get additional information
        with VideoFileClip(video_path) as clip:
            # Get more accurate duration
            duration = clip.duration
        
        metadata = {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
            'aspect_ratio': f"{width}:{height}",
            'file_size': os.path.getsize(video_path),
        }
        
        logger.info(f"Video analysis complete: {width}x{height}, {fps} fps, {duration:.2f} seconds")
        return metadata
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise


def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """
    Detect scene changes in a video.
    
    Args:
        video_path: Path to the input video file.
        
    Returns:
        List of scene time ranges as (start_time, end_time) tuples.
    """
    logger.info(f"Detecting scenes in video: {video_path}")
    
    try:
        # Use ffmpeg with scene detection
        output = subprocess.check_output([
            'ffmpeg', '-i', video_path, '-filter:v', 
            'select=\'gt(scene,0.3)\'', '-f', 'null', '-'
        ], stderr=subprocess.PIPE).decode()
        
        # Parse scene change timestamps
        scene_changes = []
        # Add scene change detection parsing logic here
        
        # For now, return a placeholder
        metadata = analyze_video(video_path)
        scene_changes = [(0, metadata['duration'])]
        
        return scene_changes
        
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise


def segment_by_speech_pauses(video_path: str, audio_segments: List[dict]) -> List[dict]:
    """
    Segment video based on speech pauses.
    
    Args:
        video_path: Path to the input video file.
        audio_segments: List of audio segment dictionaries with start and end times.
        
    Returns:
        List of video segment dictionaries with start and end times.
    """
    segments = []
    
    # Add segmentation logic based on speech pauses and audio segments
    for i, segment in enumerate(audio_segments):
        segments.append({
            'id': i,
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['end'] - segment['start'],
            'audio_segment': segment
        })
    
    return segments


def extract_frame(video_path: str, timestamp: float, output_path: str) -> str:
    """
    Extract a single frame from a video at the specified timestamp.
    
    Args:
        video_path: Path to the input video file.
        timestamp: Time in seconds to extract the frame from.
        output_path: Path to save the extracted frame.
        
    Returns:
        Path to the extracted frame.
    """
    try:
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(output_path, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except Exception as e:
        logger.error(f"Error extracting frame at {timestamp}: {e}")
        raise