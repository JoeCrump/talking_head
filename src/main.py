#!/usr/bin/env python3
"""
Main entry point for the Automatic Video Summarizer.
"""

import argparse
import logging
import os
import time
from pathlib import Path

from src import video_processing
from src import audio_processing
from src import speech_to_text
from src import content_selection
from src import script_generation
from src import voiceprint_identification
from src import video_editing

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate short-form videos from long-form content.')

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path for the output video file(s)')
    parser.add_argument('--duration', '-d', type=int, default=settings.DEFAULT_DURATION,
                        help=f'Target duration in seconds for each output video (default: {settings.DEFAULT_DURATION})')
    parser.add_argument('--aspect-ratio', '-a', type=str, default=settings.DEFAULT_ASPECT_RATIO,
                        help=f'Output aspect ratio (default: {settings.DEFAULT_ASPECT_RATIO})')
    parser.add_argument('--filler-words', type=str, default=','.join(settings.FILLER_WORDS),
                        help=f'Comma-separated list of filler words to remove')
    parser.add_argument('--save-transcript', action='store_true',
                        help='Save the full transcript to a file')
    parser.add_argument('--save-voiceprint', action='store_true',
                        help='Save the speaker voiceprint for future use')

    # Add new arguments
    parser.add_argument('--num-videos', type=int, default=1,
                        help='Number of short videos to create (max 5)')
    parser.add_argument('--add-captions', action='store_true', default=True,
                        help='Add captions to the videos')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate num_videos (between 1 and 5)
    args.num_videos = max(1, min(5, args.num_videos))

    start_time = time.time()
    logger.info(f"Starting processing of {args.input}")
    logger.info(f"Will create {args.num_videos} video(s) with{'out' if not args.add_captions else ''} captions")

    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Process video and extract audio
        logger.info("Extracting audio and processing video...")
        video_info = video_processing.analyze_video(args.input)
        audio_file = audio_processing.extract_audio(args.input)

        # Transcribe speech to text
        logger.info("Transcribing audio to text...")
        transcript = speech_to_text.transcribe(audio_file, include_timestamps=True)

        if args.save_transcript:
            transcript_path = os.path.splitext(args.output)[0] + "_transcript.txt"
            speech_to_text.save_transcript(transcript, transcript_path)
            logger.info(f"Transcript saved to {transcript_path}")

        # Extract voice profile if requested
        if args.save_voiceprint:
            logger.info("Extracting speaker voiceprint...")
            voiceprint = voiceprint_identification.extract_voiceprint(audio_file)
            voiceprint_path = os.path.splitext(args.output)[0] + "_voiceprint.npz"
            voiceprint_identification.save_voiceprint(voiceprint, voiceprint_path)
            logger.info(f"Voiceprint saved to {voiceprint_path}")

        # Clean transcript by removing filler words
        logger.info("Cleaning transcript and removing filler words...")
        filler_words = args.filler_words.split(',')
        clean_transcript = audio_processing.remove_filler_words(transcript, filler_words)

        # Select the most important content
        logger.info("Selecting key content for short video...")
        selected_content = content_selection.select_key_moments(clean_transcript, target_duration=args.duration)

        # Generate a script for the short video
        logger.info("Generating script for short video...")
        script = script_generation.generate_script(selected_content, target_duration=args.duration)

        # Create the final edited video(s)
        logger.info(f"Editing final video(s) ({args.num_videos})...")

        if args.num_videos == 1:
            # Single video mode - use original function for backward compatibility
            video_path = video_editing.create_short_video(
                input_path=args.input,
                output_path=args.output,
                script=script,
                add_captions=args.add_captions,
                aspect_ratio=args.aspect_ratio
            )
            output_videos = [video_path]
        else:
            # Multiple videos mode - use the new function
            # Build output directory based on the provided output path
            base_output_path = os.path.splitext(args.output)[0]
            output_dir = os.path.dirname(base_output_path)

            output_videos = video_editing.create_multiple_short_videos(
                input_path=args.input,
                output_dir=output_dir,
                script=script,
                num_videos=args.num_videos,
                target_duration=args.duration,
                add_captions=args.add_captions,
                aspect_ratio=args.aspect_ratio
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

        for i, video_path in enumerate(output_videos):
            logger.info(f"Short video {i + 1}/{len(output_videos)} saved to: {video_path}")

        return 0

    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        return 1


if __name__ == "__main__":
    exit(main())