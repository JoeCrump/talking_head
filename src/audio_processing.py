"""
Audio processing module for extracting audio, detecting speech, and removing filler words.
"""

import logging
import os
import tempfile
from typing import List, Dict, Any, Tuple

import ffmpeg
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import spacy

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


def detect_speech_segments(audio_path: str, min_silence_len: int = 500, silence_thresh: int = -32) -> List[Dict[str, float]]:
    """
    Detect speech segments in an audio file.
    
    Args:
        audio_path: Path to the input audio file.
        min_silence_len: Minimum length of silence in milliseconds.
        silence_thresh: Silence threshold in dB.
        
    Returns:
        List of speech segments as dictionaries with start and end times.
    """
    logger.info(f"Detecting speech segments in audio: {audio_path}")
    
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Find non-silent chunks
        non_silent_chunks = detect_nonsilent(
            audio, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        # Convert to seconds
        segments = []
        for i, (start, end) in enumerate(non_silent_chunks):
            segments.append({
                'id': i,
                'start': start / 1000.0,
                'end': end / 1000.0,
                'duration': (end - start) / 1000.0
            })
        
        logger.info(f"Found {len(segments)} speech segments")
        return segments
    
    except Exception as e:
        logger.error(f"Error detecting speech segments: {e}")
        raise


def remove_filler_words(transcript: List[Dict[str, Any]], filler_words: List[str]) -> List[Dict[str, Any]]:
    """
    Remove filler words from a transcript.

    Args:
        transcript: List of transcript segments with text and timestamps.
        filler_words: List of filler words to remove.

    Returns:
        Updated transcript with filler words removed.
    """
    logger.info("Removing filler words from transcript")

    # Basic fallback if library issues occur
    if not transcript:
        return []

    # Defensive import to prevent name shadowing
    import spacy as spacy_lib

    # Load spaCy for linguistic analysis
    try:
        nlp = spacy_lib.load("en_core_web_sm")
    except OSError:
        # Download the model if it's not available
        import spacy.cli
        spacy_lib.cli.download("en_core_web_sm")
        nlp = spacy_lib.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Could not load spacy model: {e}")
        # Fallback to simple word replacement
        cleaned_transcript = []
        for segment in transcript:
            if 'text' in segment:
                text = segment['text']
                for filler in filler_words:
                    text = text.replace(f" {filler} ", " ")
                updated_segment = segment.copy()
                updated_segment['text'] = text
                cleaned_transcript.append(updated_segment)
            else:
                cleaned_transcript.append(segment)
        return cleaned_transcript

    # Prepare filler word patterns
    filler_patterns = [w.lower().strip() for w in filler_words]

    # Process each transcript segment
    cleaned_transcript = []

    for segment in transcript:
        if 'text' not in segment:
            cleaned_transcript.append(segment)
            continue

        text = segment['text']
        doc = nlp(text)

        # Build a new text without filler words
        new_text_parts = []
        for sent in doc.sents:
            clean_sent = []
            i = 0

            while i < len(sent):
                token = sent[i]
                span_text = token.text.lower()

                # Check if this token starts a filler phrase
                is_filler = False
                for filler in filler_patterns:
                    filler_tokens = filler.split()
                    if len(filler_tokens) > 1:
                        # Multi-word filler
                        if i + len(filler_tokens) <= len(sent):
                            potential_match = ' '.join([sent[i + j].text.lower() for j in range(len(filler_tokens))])
                            if potential_match == filler:
                                is_filler = True
                                i += len(filler_tokens) - 1
                                break
                    else:
                        # Single-word filler
                        if span_text == filler:
                            is_filler = True
                            break

                if not is_filler:
                    clean_sent.append(token.text_with_ws)

                i += 1

            new_text_parts.append(''.join(clean_sent).strip())

        new_text = ' '.join(new_text_parts)

        # Update the transcript segment
        updated_segment = segment.copy()
        updated_segment['text'] = new_text
        updated_segment['original_text'] = text
        cleaned_transcript.append(updated_segment)

    logger.info("Filler words removed from transcript")
    return cleaned_transcript