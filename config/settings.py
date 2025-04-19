"""
Configuration settings for the Automatic Video Summarizer.
"""

# Default settings
DEFAULT_DURATION = 60  # Target duration in seconds
DEFAULT_ASPECT_RATIO = "16:9"  # Default aspect ratio for output videos

# Filler words to remove
FILLER_WORDS = [
    "um", "uh", "er", "ah",
    "like", "you know", "I mean",
    "sort of", "kind of", "basically",
    "actually", "literally", "so",
    "just", "well", "right", "I guess",
    "okay", "ok", "so yeah", "anyway"
]

# Content selection settings
CONTENT_IMPORTANCE_THRESHOLD = 0.6  # Threshold for considering content important (0-1)
MIN_SEGMENT_DURATION = 1.0  # Minimum duration for a segment in seconds
MAX_SEGMENT_DURATION = 10.0  # Maximum duration for a segment in seconds

# Video editing settings
VIDEO_TRANSITION_TYPE = "crossfade"  # Type of transition between clips
VIDEO_TRANSITION_DURATION = 0.3  # Duration of transitions in seconds

# Audio settings
AUDIO_FADE_IN_DURATION = 0.2  # Duration of audio fade in in seconds
AUDIO_FADE_OUT_DURATION = 0.3  # Duration of audio fade out in seconds

# Output settings
OUTPUT_FPS = 30  # Frames per second for output video
OUTPUT_VIDEO_CODEC = "libx264"  # Video codec for output
OUTPUT_AUDIO_CODEC = "aac"  # Audio codec for output
OUTPUT_VIDEO_BITRATE = "2000k"  # Bitrate for output video
OUTPUT_AUDIO_BITRATE = "128k"  # Bitrate for output audio

# API Keys
# Replace these with your actual API keys
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
DEEPGRAM_API_KEY = "YOUR_DEEPGRAM_API_KEY"
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"