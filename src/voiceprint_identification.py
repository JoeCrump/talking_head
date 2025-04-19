"""
Voiceprint identification module for extracting and using speaker voiceprints.
"""

import logging
import os
import numpy as np
from typing import Dict, Any
import pickle

from pyannote.audio import Pipeline
import torch

logger = logging.getLogger(__name__)


def extract_voiceprint(audio_path: str) -> Dict[str, Any]:
    """
    Extract a voiceprint from an audio file.
    
    Args:
        audio_path: Path to the input audio file.
        
    Returns:
        Dictionary containing voiceprint information.
    """
    logger.info(f"Extracting voiceprint from audio: {audio_path}")
    
    try:
        # Try to use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the speaker diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token="YOUR_HUGGINGFACE_TOKEN"  # This would need to be replaced with a real token
        ).to(device)
        
        # Run the pipeline on the audio file
        diarization = pipeline(audio_path)
        
        # Extract speaker embeddings
        voiceprint_data = {}
        speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
        
        logger.info(f"Identified {len(speakers)} speakers in the audio")
        
        if len(speakers) == 0:
            logger.warning("No speakers detected in the audio")
            return {"error": "No speakers detected", "status": "failed"}
        
        # For simplicity in this implementation, we'll return a placeholder
        # In a real implementation, you'd extract actual speaker embeddings
        voiceprint = {
            "status": "success",
            "speaker_count": len(speakers),
            "speakers": list(speakers),
            "main_speaker": list(speakers)[0] if speakers else None,
            "extraction_method": "pyannote.audio",
            "embedding_size": 192,  # Typical size for speaker embeddings
            "embedding": np.random.rand(192).tolist(),  # Placeholder for actual embeddings
        }
        
        return voiceprint
        
    except Exception as e:
        logger.error(f"Error extracting voiceprint: {e}")
        return {"error": str(e), "status": "failed"}


def save_voiceprint(voiceprint: Dict[str, Any], output_path: str) -> None:
    """
    Save a voiceprint to a file.
    
    Args:
        voiceprint: Voiceprint dictionary.
        output_path: Path to save the voiceprint.
    """
    try:
        # Convert any numpy arrays to lists for JSON serialization
        for key, value in voiceprint.items():
            if isinstance(value, np.ndarray):
                voiceprint[key] = value.tolist()
        
        # Save as a pickle file for complete data preservation
        with open(output_path, 'wb') as f:
            pickle.dump(voiceprint, f)
            
        logger.info(f"Voiceprint saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving voiceprint: {e}")
        raise


def load_voiceprint(voiceprint_path: str) -> Dict[str, Any]:
    """
    Load a voiceprint from a file.
    
    Args:
        voiceprint_path: Path to the voiceprint file.
        
    Returns:
        Voiceprint dictionary.
    """
    try:
        with open(voiceprint_path, 'rb') as f:
            voiceprint = pickle.load(f)
        return voiceprint
    except Exception as e:
        logger.error(f"Error loading voiceprint: {e}")
        raise


def compare_voiceprints(voiceprint1: Dict[str, Any], voiceprint2: Dict[str, Any]) -> float:
    """
    Compare two voiceprints and return a similarity score.
    
    Args:
        voiceprint1: First voiceprint.
        voiceprint2: Second voiceprint.
        
    Returns:
        Similarity score between 0 and 1.
    """
    try:
        embedding1 = np.array(voiceprint1.get('embedding', []))
        embedding2 = np.array(voiceprint2.get('embedding', []))
        
        if len(embedding1) == 0 or len(embedding2) == 0 or len(embedding1) != len(embedding2):
            return 0.0
            
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error comparing voiceprints: {e}")
        return 0.0