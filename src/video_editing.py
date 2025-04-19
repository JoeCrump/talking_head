"""
Video editing module for creating the final short-form videos with captions.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
logger = logging.getLogger(__name__)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut


def create_multiple_short_videos(
    input_path: str,
    output_dir: str,
    script: Dict[str, Any],
    num_videos: int = 1,
    target_duration: int = 60,
    add_captions: bool = True,
    aspect_ratio: str = "16:9"
) -> List[str]:
    """
    Create multiple short videos from the input video based on the script.

    Args:
        input_path: Path to the input video file.
        output_dir: Directory to save output videos.
        script: Script dictionary with segment information.
        num_videos: Number of short videos to create (max 5).
        target_duration: Target duration for each video in seconds.
        add_captions: Whether to add captions to the videos.
        aspect_ratio: Output aspect ratio (e.g., "16:9", "9:16").

    Returns:
        List of paths to the created video files.
    """
    # Validate num_videos
    num_videos = max(1, min(5, num_videos))
    logger.info(f"Creating {num_videos} short videos from {input_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Divide script segments into groups for multiple videos
    segment_groups = _divide_segments_for_multiple_videos(script['segments'], num_videos)

    output_paths = []

    # Create each video
    for i, segments in enumerate(segment_groups):
        # Create sub-script for this video
        sub_script = {
            'title': f"{script.get('title', 'Short Video')} (Part {i+1})",
            'segments': segments
        }

        # Set output path for this video
        output_filename = f"short_video_{i+1}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Create the short video
        try:
            create_short_video(
                input_path,
                output_path,
                sub_script,
                add_captions=add_captions,
                aspect_ratio=aspect_ratio
            )
            output_paths.append(output_path)
            logger.info(f"Created video {i+1}/{num_videos}: {output_path}")
        except Exception as e:
            logger.error(f"Error creating video {i+1}: {e}")

    return output_paths


def _divide_segments_for_multiple_videos(segments: List[Dict[str, Any]], num_videos: int) -> List[List[Dict[str, Any]]]:
    """
    Divide segments into logical groups for multiple videos.

    Args:
        segments: List of segment dictionaries.
        num_videos: Number of videos to create.

    Returns:
        List of segment groups, one for each video.
    """
    if num_videos <= 1:
        return [segments]

    # Filter out segments without proper timing info
    valid_segments = [s for s in segments if 'start' in s and 'end' in s]

    if not valid_segments:
        return [[]]

    # Sort by start time
    valid_segments.sort(key=lambda x: x['start'])

    # Calculate total duration
    total_duration = sum(s['end'] - s['start'] for s in valid_segments)
    target_duration_per_video = total_duration / num_videos

    # Create segment groups
    segment_groups = [[] for _ in range(num_videos)]
    current_group = 0
    current_duration = 0

    for segment in valid_segments:
        segment_duration = segment['end'] - segment['start']

        # If adding this segment exceeds target duration and we're not at the last group,
        # move to the next group
        if (current_duration + segment_duration > target_duration_per_video and
            current_group < num_videos - 1):
            current_group += 1
            current_duration = 0

        # Add segment to current group
        segment_groups[current_group].append(segment)
        current_duration += segment_duration

    # Ensure we don't have empty groups
    return [group for group in segment_groups if group]


def create_short_video(
    input_path: str,
    output_path: str,
    script: Dict[str, Any],
    add_captions: bool = True,
    aspect_ratio: str = "16:9"
) -> str:
    """
    Create a short video from the input video based on the script.

    Args:
        input_path: Path to the input video file.
        output_path: Path to save the output video.
        script: Script dictionary with segment information.
        add_captions: Whether to add captions to the video.
        aspect_ratio: Output aspect ratio (e.g., "16:9", "9:16").

    Returns:
        Path to the created video file.
    """
    logger.info(f"Creating short video from {input_path} with {len(script['segments'])} segments")

    try:
        # Load the input video
        with VideoFileClip(input_path) as video:
            # Extract segments from the script
            segments_to_use = []

            for segment in script['segments']:
                if 'start' in segment and 'end' in segment:
                    start_time = segment['start']
                    end_time = segment['end']

                    # Skip segments with invalid times
                    if start_time < 0 or end_time <= start_time or end_time > video.duration:
                        logger.warning(f"Skipping segment with invalid times: {start_time}-{end_time}")
                        continue

                    segments_to_use.append((start_time, end_time, segment))

            # Sort segments by start time
            segments_to_use.sort(key=lambda x: x[0])

            logger.info(f"Processing {len(segments_to_use)} valid segments")

            # Extract clips for each segment
            clips = []
            for start_time, end_time, segment in segments_to_use:
                clip = video.subclipped(start_time, end_time)  # Changed from subclipped to subclip

                # Add captions if enabled and text is available
                # Before your if/elif block, initialize final_clip
                final_clip = clip  # Default to the original clip

                if add_captions and 'text' in segment:
                    # Create text clip with ALL required parameters
                    txt_clip = TextClip(
                        font='/usr/share/fonts/open-sans/OpenSans-Regular.ttf',
                        text=segment['text'],
                        font_size=30,
                        color='white',
                        bg_color=(0, 0, 0, 128),
                        size=(int(clip.w * 0.9), None),
                        method='label',  # Changed from 'caption' to 'label'
                        text_align='center',  # Explicitly set alignment
                        horizontal_align='center',
                        vertical_align='center'
                    )

                    # Set duration
                    txt_clip.duration = clip.duration

                    # Set position
                    txt_clip.pos = lambda t: ('center', clip.h * 0.85)

                    # Create composite clip
                    final_clip = CompositeVideoClip([clip, txt_clip], size=clip.size)
                    final_clip.duration = clip.duration  # Ensure the duration is properly set

                # Or add any specific overlay text if provided
                elif 'overlay_text' in segment:
                    # Create text clip
                    txt_clip = TextClip(
                        font='/usr/share/fonts/open-sans/OpenSans-Regular.ttf',
                        text=segment['text'],
                        font_size=30,
                        color='white',
                        bg_color=(0, 0, 0, 128),
                        size=(int(clip.w * 0.9), None),
                        method='label',  # Changed from 'caption' to 'label'
                        text_align='center',  # Explicitly set alignment
                        horizontal_align='center',
                        vertical_align='center'
                    )

                    # Set duration
                    txt_clip.duration = clip.duration

                    # Set position
                    txt_clip.pos = lambda t: ('center', clip.h * 0.85)

                    # Create composite clip
                    final_clip = CompositeVideoClip([clip, txt_clip], size=clip.size)
                    final_clip.duration = clip.duration  # Ensure the duration is properly set

                # Now final_clip is guaranteed to be defined
                clips.append(final_clip)

            if not clips:
                logger.error("No valid clips to include in the output video")
                return None

            # Add transitions between clips
            if len(clips) > 1:
                clips = add_transitions(clips, transition_type="crossfade", transition_duration=0.5)

            # Concatenate the clips
            final_clip = concatenate_videoclips(clips, method="compose")

            # Apply the desired aspect ratio
            final_clip = _adjust_aspect_ratio(final_clip, aspect_ratio)

            # Write the final clip to the output file
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=f"{output_path}.temp-audio.m4a",
                remove_temp=True,
                fps=24
            )

            logger.info(f"Short video created: {output_path}, duration: {final_clip.duration:.2f}s")
            return output_path

    except Exception as e:
        logger.error(f"Error creating short video: {e}")
        raise


def _adjust_aspect_ratio(clip, aspect_ratio: str) -> VideoFileClip:
    """Adjust the aspect ratio of a video clip."""
    # Parse the aspect ratio
    try:
        width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
        target_aspect = width_ratio / height_ratio
    except (ValueError, ZeroDivisionError):
        logger.warning(f"Invalid aspect ratio: {aspect_ratio}, using original")
        return clip

    # Calculate the current aspect ratio
    current_aspect = clip.w / clip.h

    # If they're close enough, no need to adjust
    if abs(current_aspect - target_aspect) < 0.01:
        return clip

    # Determine if we need to crop width or height
    if current_aspect > target_aspect:
        # Current video is wider than target, crop the width
        new_width = int(clip.h * target_aspect)
        x_center = clip.w // 2
        return clip.crop(x1=x_center - new_width // 2,
                          y1=0,
                          width=new_width,
                          height=clip.h)
    else:
        # Current video is taller than target, crop the height
        new_height = int(clip.w / target_aspect)
        y_center = clip.h // 2
        return clip.crop(x1=0,
                          y1=y_center - new_height // 2,
                          width=clip.w,
                          height=new_height)


def add_transitions(clips: List[VideoFileClip], transition_type: str = "crossfade", transition_duration: float = 0.5) -> List[VideoFileClip]:
    """
    Add transitions between video clips.

    Args:
        clips: List of video clips.
        transition_type: Type of transition to add.
        transition_duration: Duration of transition in seconds.

    Returns:
        List of clips with transitions added.
    """
    if len(clips) < 2:
        return clips

    result_clips = []

    for i in range(len(clips)):
        clip = clips[i]

        # Apply CrossFadeIn to all clips except the first one
        if i > 0 and transition_type == "crossfade":
            # Make sure we don't try to crossfade longer than the clip duration
            safe_duration = min(transition_duration, clip.duration * 0.5)
            if safe_duration > 0:
                # Create and apply the CrossFadeIn effect
                fade_in = CrossFadeIn(safe_duration)
                clip = fade_in.copy().apply(clip)

        # Apply CrossFadeOut to all clips except the last one
        if i < len(clips) - 1 and transition_type == "crossfade":
            # Make sure we don't try to crossfade longer than the clip duration
            safe_duration = min(transition_duration, clip.duration * 0.5)
            if safe_duration > 0:
                # Create and apply the CrossFadeOut effect
                fade_out = CrossFadeOut(safe_duration)  # Fixed: removed the module prefix
                clip = fade_out.copy().apply(clip)

        result_clips.append(clip)

    return result_clips