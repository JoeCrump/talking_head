"""
Script generation module for creating a coherent script from selected content.
"""

import logging
import os
from typing import List, Dict, Any
import json
import openai

logger = logging.getLogger(__name__)


def generate_script(selected_content: List[Dict[str, Any]], target_duration: int = 60) -> Dict[str, Any]:
    """
    Generate a script for the short video from selected content.
    
    Args:
        selected_content: List of selected transcript segments.
        target_duration: Target duration of the output video in seconds.
        
    Returns:
        Dictionary containing script information including segments and metadata.
    """
    logger.info(f"Generating script for {target_duration}s short video")
    
    # First approach: use the selected content directly
    direct_script = _create_direct_script(selected_content)
    
    # Calculate if we need to regenerate or refine the script
    direct_duration = sum(
        segment.get('end', 0) - segment.get('start', 0) 
        for segment in direct_script['segments'] 
        if 'start' in segment and 'end' in segment
    )
    
    logger.info(f"Direct script duration: {direct_duration:.2f}s (target: {target_duration}s)")
    
    # If direct script is within 20% of target duration, use it
    if 0.8 * target_duration <= direct_duration <= 1.2 * target_duration:
        logger.info("Using direct script (within target duration range)")
        return direct_script
    
    # Otherwise, try to refine the script using AI
    try:
        refined_script = _refine_script_with_ai(selected_content, direct_script, target_duration)
        return refined_script
    except Exception as e:
        logger.error(f"Error refining script with AI: {e}")
        logger.info("Falling back to direct script")
        return direct_script


def _create_direct_script(selected_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a script directly from selected content without modifications."""
    script = {
        'title': 'Short Video Script',
        'segments': selected_content.copy(),
        'metadata': {
            'generated_by': 'direct_selection',
            'segment_count': len(selected_content)
        }
    }
    
    # Add script formatting
    for i, segment in enumerate(script['segments']):
        if 'text' in segment:
            segment['script_text'] = segment['text']
            
    return script


def _refine_script_with_ai(selected_content: List[Dict[str, Any]], direct_script: Dict[str, Any], target_duration: int) -> Dict[str, Any]:
    """Refine the script using OpenAI to better fit target duration and improve coherence."""
    logger.info("Refining script with AI")
    
    # Combine the text from selected content
    combined_text = " ".join(segment['text'] for segment in selected_content if 'text' in segment)
    
    # Create a prompt for OpenAI
    prompt = f"""
    You are an expert video editor tasked with creating a concise, engaging short-form video script.
    
    Original content: "{combined_text}"
    
    Please create a script for a {target_duration}-second video that:
    1. Captures the most important information from the original content
    2. Flows naturally and is engaging
    3. Is appropriate for short-form video platforms
    4. Removes any redundancies or unnecessary details
    5. Maintains the speaker's original voice and style
    
    Format your response as JSON with:
    1. A title field
    2. A segments array containing objects with script_text fields
    """
    
    # Call OpenAI API
    try:
        openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert video editor and scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        
        # Parse the response
        ai_content = response.choices[0].message.content.strip()
        
        # Try to extract JSON content if surrounded by markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', ai_content)
        if json_match:
            ai_content = json_match.group(1)
        
        try:
            ai_script = json.loads(ai_content)
        except json.JSONDecodeError:
            logger.warning("Could not parse AI response as JSON. Using direct script.")
            return direct_script
        
        # Make sure the AI script has the required fields
        if not isinstance(ai_script, dict) or 'segments' not in ai_script or not isinstance(ai_script['segments'], list):
            logger.warning("AI response doesn't have the expected format. Using direct script.")
            return direct_script
        
        # Merge AI script with timing information from direct script
        refined_script = ai_script.copy()
        
        # Try to align the AI segments with the original timing information
        if len(refined_script['segments']) <= len(direct_script['segments']):
            # If AI produced fewer or equal segments, use original timing
            for i, segment in enumerate(refined_script['segments']):
                if i < len(direct_script['segments']):
                    # Copy timing information
                    for key in ['start', 'end', 'id']:
                        if key in direct_script['segments'][i]:
                            segment[key] = direct_script['segments'][i][key]
        else:
            # AI produced more segments than we have timing for
            # Distribute the original segments' timing across the AI segments
            total_ai_segments = len(refined_script['segments'])
            total_original_duration = sum(
                segment['end'] - segment['start'] 
                for segment in direct_script['segments'] 
                if 'start' in segment and 'end' in segment
            )
            
            # Calculate segment duration based on character count
            total_chars = sum(len(segment.get('script_text', '')) for segment in refined_script['segments'])
            
            start_time = direct_script['segments'][0].get('start', 0) if direct_script['segments'] else 0
            for i, segment in enumerate(refined_script['segments']):
                segment_chars = len(segment.get('script_text', ''))
                segment_duration = (segment_chars / total_chars) * total_original_duration
                
                segment['id'] = i
                segment['start'] = start_time
                segment['end'] = start_time + segment_duration
                
                start_time += segment_duration
        
        # Add metadata
        refined_script['metadata'] = {
            'generated_by': 'ai_refinement',
            'segment_count': len(refined_script['segments']),
            'target_duration': target_duration
        }
        
        return refined_script
        
    except Exception as e:
        logger.error(f"Error generating refined script: {e}")
        return direct_script