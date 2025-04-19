"""
Speech-to-text module for transcribing audio with timestamps.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import json
import tempfile
import requests
import time
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Constants for chunking
CHUNK_LENGTH_S = 300  # 5-minute chunks for OpenAI API
LARGE_FILE_THRESHOLD_MB = 20  # Files larger than this will be chunked

def transcribe(audio_path: str, include_timestamps: bool = True) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file to text with timestamps.

    Args:
        audio_path: Path to the input audio file.
        include_timestamps: Whether to include timestamps in the transcript.

    Returns:
        List of transcript segments with text and timestamps.
    """
    logger.info(f"Transcribing audio: {audio_path}")

    try:
        # Select method based on file size
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        if file_size_mb > LARGE_FILE_THRESHOLD_MB:
            logger.info(f"Large audio file detected ({file_size_mb:.1f}MB). Using chunked processing.")
            return _transcribe_with_openai_chunked(audio_path, include_timestamps)
        else:
            logger.info(f"Standard audio file ({file_size_mb:.1f}MB). Using regular processing.")
            return _transcribe_with_openai(audio_path, include_timestamps)

    except Exception as e:
        logger.error(f"Error during transcription: {e}")

        # Check if it's an API key error
        if "api_key" in str(e).lower() or "apikey" in str(e).lower():
            logger.warning("API key issue detected. Falling back to transformers Whisper.")
            # Import and use transformers here as a fallback
            from transformers import pipeline
            return _transcribe_with_transformers(audio_path, include_timestamps)

        # Fallback to transformers pipeline
        try:
            logger.info("Falling back to transformers Whisper")
            from transformers import pipeline
            return _transcribe_with_transformers(audio_path, include_timestamps)
        except Exception as e2:
            logger.error(f"Error with fallback transcription: {e2}")
            # Return empty transcript rather than raising an exception
            return []


def _transcribe_with_openai(audio_path: str, include_timestamps: bool) -> List[Dict[str, Any]]:
    """Transcribe audio using OpenAI Whisper API."""
    logger.info("Using OpenAI Whisper API for transcription")

    try:
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Create headers for API request
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Prepare file for upload
        with open(audio_path, "rb") as audio_file:
            # Make API request
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": audio_file},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"] if include_timestamps else []
                }
            )

        # Check for errors
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")

        # Parse response
        result = response.json()

        # Format the result
        transcript = []

        if include_timestamps and "segments" in result:
            # OpenAI returns segments with timestamps
            for i, segment in enumerate(result["segments"]):
                transcript.append({
                    'id': i,
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                })
        else:
            # No timestamps, just text
            transcript.append({
                'id': 0,
                'text': result['text'].strip(),
            })

        logger.info(f"Transcription complete: {len(transcript)} segments")
        return transcript

    except Exception as e:
        logger.error(f"Error in OpenAI Whisper transcription: {e}")
        raise


def _transcribe_with_openai_chunked(audio_path: str, include_timestamps: bool) -> List[Dict[str, Any]]:
    """Process large audio files by splitting them into smaller chunks."""
    logger.info("Using chunked OpenAI Whisper processing")

    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(audio_path)

        # Get total duration in milliseconds
        total_duration_ms = len(audio)
        chunk_length_ms = CHUNK_LENGTH_S * 1000  # Convert seconds to ms

        # Calculate number of chunks
        num_chunks = (total_duration_ms + chunk_length_ms - 1) // chunk_length_ms
        logger.info(f"Processing {num_chunks} chunks of audio")

        # Process each chunk
        all_segments = []
        time_offset = 0

        for i in range(num_chunks):
            chunk_start = i * chunk_length_ms
            chunk_end = min((i + 1) * chunk_length_ms, total_duration_ms)

            logger.info(f"Processing chunk {i+1}/{num_chunks} ({chunk_start/1000:.1f}s - {chunk_end/1000:.1f}s)")

            # Extract chunk
            chunk = audio[chunk_start:chunk_end]

            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"chunk_{i}.wav")

            try:
                # Export chunk to temp file
                chunk.export(temp_path, format="wav")

                # Transcribe chunk
                chunk_segments = _transcribe_with_openai(temp_path, include_timestamps)

                # Process results - add time offset to each segment
                for segment in chunk_segments:
                    if 'start' in segment and 'end' in segment:
                        segment['start'] += (chunk_start / 1000)  # Convert ms to seconds
                        segment['end'] += (chunk_start / 1000)
                    else:
                        segment['start'] = chunk_start / 1000
                        segment['end'] = chunk_end / 1000

                    segment['id'] = len(all_segments)
                    all_segments.append(segment)

            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i+1}: {chunk_error}")

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

                # Add a small delay between API calls to avoid rate limits
                if i < num_chunks - 1:
                    time.sleep(0.5)

        # Sort segments by start time
        all_segments.sort(key=lambda x: x.get('start', 0))

        logger.info(f"Chunked transcription complete: {len(all_segments)} segments")
        return all_segments

    except Exception as e:
        logger.error(f"Error in chunked transcription: {e}")
        raise


def _transcribe_with_transformers(audio_path: str, include_timestamps: bool) -> List[Dict[str, Any]]:
    """Transcribe using Whisper via transformers (fallback method)."""
    logger.info("Using transformers Whisper for transcription")

    try:
        # Import here to avoid dependency if not needed
        from transformers import pipeline
        import torch

        # Select device
        device = 0 if torch.cuda.is_available() else -1

        # Get model size from environment, default to tiny for Heroku
        model_name = os.environ.get("WHISPER_MODEL", "openai/whisper-tiny")

        # Initialize the Whisper pipeline
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            stride_length_s=5,
            device=device,
        )

        # Generate transcription
        result = whisper_pipeline(
            audio_path,
            return_timestamps=include_timestamps,
        )

        # Format the result
        transcript = []

        if include_timestamps and 'chunks' in result:
            # Whisper returns chunks with timestamps
            for i, chunk in enumerate(result['chunks']):
                # Make sure timestamps are valid
                if chunk['timestamp'] and len(chunk['timestamp']) >= 2:
                    transcript.append({
                        'id': i,
                        'start': chunk['timestamp'][0],
                        'end': chunk['timestamp'][1],
                        'text': chunk['text'].strip(),
                    })
        else:
            # No timestamps, just text
            transcript.append({
                'id': 0,
                'text': result['text'].strip(),
            })

        logger.info(f"Transcription complete: {len(transcript)} segments")
        return transcript

    except Exception as e:
        logger.error(f"Error in transformers Whisper transcription: {e}")
        return []


def save_transcript(transcript: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save transcript to a file.

    Args:
        transcript: List of transcript segments.
        output_path: Path to save the transcript.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Save as formatted text
            f.write("Transcript:\n\n")

            for segment in transcript:
                if 'start' in segment and 'end' in segment:
                    time_str = f"[{_format_time(segment['start'])} - {_format_time(segment['end'])}]"
                    f.write(f"{time_str} {segment['text']}\n")
                else:
                    f.write(f"{segment['text']}\n")

        # Also save as JSON for programmatic access
        json_path = output_path.replace('.txt', '.json')
        if json_path == output_path:
            json_path += '.json'

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcript saved to {output_path} and {json_path}")

    except Exception as e:
        logger.error(f"Error saving transcript: {e}")
        raise


def _format_time(seconds: float) -> str:
    """Format time in seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"