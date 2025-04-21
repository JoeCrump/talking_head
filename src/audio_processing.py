"""
Audio processing module for extracting audio and removing filler words.
"""
import logging
import os
import tempfile
import re
from typing import List, Dict, Any

import ffmpeg

logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the extracted audio file. If None, creates a temporary file.

    Returns:
        Path to the extracted audio file.
    """
    logger.info(f"Extracting audio from video: {video_path}")

    if output_path is None:
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}.wav")

    try:
        (
            ffmpeg
            .input(video_path)
            .audio
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"Audio extracted to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise


def remove_filler_words(transcript: List[Dict[str, Any]], filler_words: List[str]) -> List[Dict[str, Any]]:
    """
    Remove filler words from a transcript using regex (lightweight version).

    Args:
        transcript: List of transcript segments with text and timestamps.
        filler_words: List of filler words to remove.

    Returns:
        Updated transcript with filler words removed.
    """
    logger.info("Removing filler words from transcript")

    if not transcript:
        return []

    # Prepare regex patterns for filler words
    patterns = []
    for word in filler_words:
        # Create pattern with word boundaries
        pattern = r'\b' + re.escape(word) + r'\b'
        patterns.append(pattern)

    # Combine patterns
    combined_pattern = '|'.join(patterns)

    # Process each segment
    cleaned_transcript = []
    for segment in transcript:
        if 'text' not in segment:
            cleaned_transcript.append(segment)
            continue

        # Store original text
        original_text = segment['text']

        # Remove filler words and normalize spaces
        cleaned_text = re.sub(combined_pattern, '', original_text)
        # Fix multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Update segment
        updated_segment = segment.copy()
        updated_segment['text'] = cleaned_text
        updated_segment['original_text'] = original_text
        cleaned_transcript.append(updated_segment)

    logger.info("Filler words removed from transcript")
    return cleaned_transcript