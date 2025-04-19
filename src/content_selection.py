"""
Content selection module for identifying key moments in a transcript.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)


def select_key_moments(transcript: List[Dict[str, Any]], target_duration: int = 60) -> List[Dict[str, Any]]:
    """
    Select key moments from the transcript to create a concise summary.
    
    Args:
        transcript: List of transcript segments with text and timestamps.
        target_duration: Target duration of the output video in seconds.
        
    Returns:
        List of selected transcript segments for the summary.
    """
    logger.info(f"Selecting key moments from transcript for {target_duration}s summary")
    
    # Check if transcript has timestamps
    has_timestamps = all('start' in segment and 'end' in segment for segment in transcript)
    if not has_timestamps:
        logger.warning("Transcript doesn't have timestamps. Using text-only selection.")
        return _select_by_text_importance(transcript, target_duration)
    
    # Calculate total duration of the transcript
    total_duration = sum(segment['end'] - segment['start'] for segment in transcript)
    
    if total_duration <= target_duration:
        logger.info(f"Original content ({total_duration}s) is shorter than target ({target_duration}s). Using full content.")
        return transcript
    
    # Selection ratio - what percentage of content to keep
    # Increased from original to ensure more content is selected
    selection_ratio = min(1.0, (target_duration / total_duration) * 2.0)
    logger.info(f"Selection ratio: {selection_ratio:.2%}")

    # Combine text-based and segment-based approaches
    # Increase target duration by 30% to ensure we get enough content
    adjusted_target = target_duration * 1.3
    selected_by_text = _select_by_text_importance(transcript, adjusted_target)
    selected_by_duration = _select_by_duration(transcript, adjusted_target)

    # Merge selections, preferring text-based approach
    selected_text_ids = {segment['id'] for segment in selected_by_text if 'id' in segment}
    final_selection = selected_by_text.copy()

    for segment in selected_by_duration:
        if 'id' in segment and segment['id'] not in selected_text_ids:
            # Only add segments from duration-based selection if they weren't in the text-based selection
            segment_duration = segment['end'] - segment['start']
            # Check if adding this would exceed target duration by more than 40%
            # Changed from 20% to 40% to allow more content
            current_duration = sum(seg['end'] - seg['start'] for seg in final_selection if 'end' in seg and 'start' in seg)
            if current_duration + segment_duration <= target_duration * 1.4:
                final_selection.append(segment)
                selected_text_ids.add(segment['id'])

    # Sort by original position
    final_selection.sort(key=lambda x: x['id'] if 'id' in x else 0)

    actual_duration = sum(segment['end'] - segment['start'] for segment in final_selection if 'end' in segment and 'start' in segment)
    logger.info(f"Selected {len(final_selection)} segments with total duration of {actual_duration:.2f}s")

    # If we still don't have enough content, select more based on duration
    if actual_duration < target_duration * 0.8 and actual_duration > 0:
        logger.warning(f"Selected content duration ({actual_duration:.2f}s) is less than 80% of target ({target_duration}s)")
        logger.info("Adding more content to reach target duration")

        # Try again with more aggressive selection
        additional = _select_additional_content(transcript, target_duration - actual_duration, selected_text_ids)
        final_selection.extend(additional)
        final_selection.sort(key=lambda x: x['id'] if 'id' in x else 0)

        actual_duration = sum(segment['end'] - segment['start'] for segment in final_selection if 'end' in segment and 'start' in segment)
        logger.info(f"After adding content: {len(final_selection)} segments with duration of {actual_duration:.2f}s")

    return final_selection


def _select_by_text_importance(transcript: List[Dict[str, Any]], target_duration: int) -> List[Dict[str, Any]]:
    """Select segments based on text importance using AI summarization."""
    logger.info("Selecting content based on text importance")

    try:
        # Use a summarization model to identify important segments
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Combine all text
        full_text = " ".join(segment['text'] for segment in transcript if 'text' in segment)

        # If text is too long for the model, chunk it
        max_length = 1024
        if len(full_text.split()) > max_length:
            chunks = _chunk_text(full_text, max_length)
            summaries = []

            for chunk in chunks:
                # Increase max_length parameter to capture more content (changed from 0.4 to 0.6)
                summary = summarizer(chunk, max_length=int(len(chunk.split()) * 0.6), min_length=20)[0]['summary_text']
                summaries.append(summary)

            combined_summary = " ".join(summaries)
        else:
            # Increase max_length parameter to capture more content (changed from 0.4 to 0.6)
            combined_summary = summarizer(full_text, max_length=int(len(full_text.split()) * 0.6), min_length=20)[0]['summary_text']

        # Now find which segments contain parts of the summary
        selected_segments = []
        summary_words = set(re.findall(r'\b\w+\b', combined_summary.lower()))

        # Score each segment by how many summary words it contains
        for segment in transcript:
            if 'text' not in segment:
                continue

            segment_words = set(re.findall(r'\b\w+\b', segment['text'].lower()))
            overlap = len(segment_words.intersection(summary_words))

            # Add a score based on word overlap
            segment_copy = segment.copy()
            segment_copy['score'] = overlap / max(1, len(segment_words))
            selected_segments.append(segment_copy)

        # Sort by score
        selected_segments.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Select top segments until we reach target duration
        # Allow up to 120% of target duration (increased from 100%)
        allowed_duration = target_duration * 1.2
        final_segments = []
        current_duration = 0

        for segment in selected_segments:
            if 'start' in segment and 'end' in segment:
                segment_duration = segment['end'] - segment['start']
                if current_duration + segment_duration <= allowed_duration:
                    final_segments.append(segment)
                    current_duration += segment_duration
            else:
                # If no timestamps, just include it
                final_segments.append(segment)

        # Sort by original position
        final_segments.sort(key=lambda x: x['id'] if 'id' in x else 0)
        return final_segments

    except Exception as e:
        logger.error(f"Error in text importance selection: {e}")
        logger.info("Falling back to duration-based selection")
        return _select_by_duration(transcript, target_duration)


def _select_by_duration(transcript: List[Dict[str, Any]], target_duration: int) -> List[Dict[str, Any]]:
    """Select segments based on evenly spaced sampling to meet target duration."""
    if not transcript:
        return []

    # Calculate total duration of input
    total_duration = 0
    segments_with_duration = []

    for segment in transcript:
        if 'start' in segment and 'end' in segment:
            duration = segment['end'] - segment['start']
            total_duration += duration
            segments_with_duration.append((segment, duration))

    if total_duration == 0:
        return transcript

    # Calculate selection ratio - multiply by 2.5 to select more content
    selection_ratio = min(1.0, (target_duration / total_duration) * 2.5)

    # Select segments proportionally
    selected = []
    current_duration = 0

    # Prioritize segments based on length (prefer longer segments)
    segments_with_duration.sort(key=lambda x: x[1], reverse=True)

    # Allow up to 130% of target duration
    max_allowed_duration = target_duration * 1.3

    for segment, duration in segments_with_duration:
        if current_duration + duration <= max_allowed_duration:
            selected.append(segment)
            current_duration += duration

    # Sort by original position
    selected.sort(key=lambda x: x['id'] if 'id' in x else 0)
    return selected


def _select_additional_content(transcript: List[Dict[str, Any]], remaining_duration: float, used_ids: set) -> List[Dict[str, Any]]:
    """Select additional segments to fill the remaining duration."""
    available_segments = []

    for segment in transcript:
        if 'id' in segment and segment['id'] not in used_ids and 'start' in segment and 'end' in segment:
            duration = segment['end'] - segment['start']
            available_segments.append((segment, duration))

    # Sort by duration (prefer segments that better fit the remaining space)
    available_segments.sort(key=lambda x: abs(remaining_duration - x[1]))

    additional = []
    current_duration = 0

    # Allow up to 140% of remaining duration
    max_allowed = remaining_duration * 1.4

    for segment, duration in available_segments:
        if current_duration + duration <= max_allowed:
            additional.append(segment)
            current_duration += duration

    return additional


def _chunk_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks of maximum length while respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_length + sentence_words <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_words
        else:
            # Start a new chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_words
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks