"""
Module for transcribing audio to text using OpenAI Whisper API with chunking for large files.
"""
import os
import logging
import tempfile
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
import ffmpeg

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_SIZE_MB = 24  # OpenAI limit is 25MB, using 24 for safety
CHUNK_LENGTH_SECONDS = 600  # 10 minutes per chunk

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def transcribe(audio_file: str, include_timestamps: bool = True) -> List[Dict[str, Any]]:
    """
    Transcribe audio file to text segments with timestamps using OpenAI Whisper API.
    Handles large files by splitting into chunks.

    Args:
        audio_file: Path to the audio file
        include_timestamps: Whether to include timestamps

    Returns:
        List of segments with text and timestamps
    """
    try:
        logger.info(f"Transcribing audio: {audio_file}")

        # Check if client is initialized
        if client is None:
            logger.error("OpenAI client not initialized. Set the OPENAI_API_KEY environment variable.")
            return []

        # Check file exists
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return []

        # Check file size
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")

        # If file is too large, split into chunks
        if file_size_mb > MAX_CHUNK_SIZE_MB:
            logger.info(f"File exceeds {MAX_CHUNK_SIZE_MB}MB limit, splitting into chunks")
            chunks = split_audio_file(audio_file)
            segments = []
            time_offset = 0

            for i, chunk in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                chunk_segments = transcribe_single_file(chunk, include_timestamps)

                # Apply time offset to timestamps
                for segment in chunk_segments:
                    segment["start"] += time_offset
                    segment["end"] += time_offset

                segments.extend(chunk_segments)

                # Update time offset for next chunk
                chunk_duration = get_audio_duration(chunk)
                time_offset += chunk_duration

            logger.info(f"Completed transcription of all {len(chunks)} chunks")
            return segments
        else:
            return transcribe_single_file(audio_file, include_timestamps)

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return []


def transcribe_single_file(audio_file: str, include_timestamps: bool = True) -> List[Dict[str, Any]]:
    """Transcribe a single audio file."""
    try:
        # Open the audio file and transcribe
        with open(audio_file, "rb") as audio:
            if include_timestamps:
                # Use timestamps option
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            else:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )

        # Format results
        segments = []

        # Try to get segments from response
        if hasattr(response, "segments"):
            for segment in response.segments:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
        # Try to access as dictionary
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
            if "segments" in response_dict:
                for segment in response_dict["segments"]:
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip()
                    })

        # If no segments found, fallback to creating a single segment
        if not segments and hasattr(response, "text"):
            segments.append({
                "start": 0,
                "end": get_audio_duration(audio_file),
                "text": response.text.strip()
            })

        return segments
    except Exception as e:
        logger.error(f"Error transcribing single file: {e}")
        return []


def split_audio_file(audio_file: str) -> List[str]:
    """Split audio file into chunks of approximately 24MB each."""
    temp_dir = tempfile.mkdtemp()
    chunks = []

    try:
        # Get audio duration
        duration = get_audio_duration(audio_file)
        num_chunks = max(1, int(duration / CHUNK_LENGTH_SECONDS) + 1)

        logger.info(f"Splitting {duration:.2f}s audio into {num_chunks} chunks")

        # Generate chunks
        for i in range(num_chunks):
            start_time = i * CHUNK_LENGTH_SECONDS
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")

            # Extract chunk using FFmpeg
            try:
                (
                    ffmpeg
                    .input(audio_file, ss=start_time, t=CHUNK_LENGTH_SECONDS)
                    .output(chunk_path, acodec='pcm_s16le', ac=1, ar='16k')
                    .overwrite_output()
                    .run(quiet=True)
                )
                chunks.append(chunk_path)
            except Exception as e:
                logger.error(f"Error splitting chunk {i}: {e}")

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error splitting audio file: {e}")
        return []


def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        probe = ffmpeg.probe(audio_file)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 30.0  # Default fallback duration