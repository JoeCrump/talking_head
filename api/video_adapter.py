"""
Adapter module to bridge between API and core video processing functions
"""
import os
import logging
from typing import List, Dict, Any

from src import video_editing

logger = logging.getLogger(__name__)

def create_short_videos(
    file_path: str,
    segments: List[Dict[str, Any]],
    num_videos: int = 5,
    add_captions: bool = True,
    task_id: str = None
) -> List[str]:
    """
    Adapter function that maps API calls to the video_editing module functions.
    """
    logger.info(f"Creating {num_videos} videos with {len(segments)} segments")

    # Create output directory using task_id
    output_dir = os.path.join("static", "output", task_id)
    os.makedirs(output_dir, exist_ok=True)

    # Format segments correctly - handle potential format issues
    processed_segments = []
    for segment in segments:
        # Handle different segment formats
        try:
            if isinstance(segment, dict):
                # Ensure we have start and end times
                if 'start' in segment and 'end' in segment:
                    # Make sure values are numeric
                    start = float(segment['start'])
                    end = float(segment['end'])

                    # Get text if available
                    text = segment.get('text', '')
                    if not text and 'content' in segment:
                        text = segment['content']

                    processed_segment = {
                        'start': start,
                        'end': end,
                        'text': text
                    }
                    processed_segments.append(processed_segment)
        except Exception as e:
            logger.error(f"Error processing segment: {e}, segment: {segment}")

    # Create script in the format expected by create_multiple_short_videos
    script = {
        'title': 'Short Video',
        'segments': processed_segments
    }

    # Call the actual function in video_editing
    try:
        output_paths = video_editing.create_multiple_short_videos(
            input_path=file_path,
            output_dir=output_dir,
            script=script,
            num_videos=num_videos,
            add_captions=add_captions
        )

        # Return just the filenames, not full paths
        return [os.path.basename(path) for path in output_paths]
    except Exception as e:
        logger.error(f"Error in create_multiple_short_videos: {e}")
        return []