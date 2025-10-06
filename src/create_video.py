import pandas as pd
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Union, List
import os
from src.config import (
    CLIPS_OUTPUT_DIR,
    VIDEO_CODEC,
    AUDIO_CODEC,
    VIDEO_BITRATE,
    CLIP_FADE_DURATION
)

def create_clip(
    video_path: Union[str, Path],
    start_time: float,
    end_time: float,
    output_path: Union[str, Path],
    fade: bool = True
) -> str:
    """
    Create a video clip from the original video using start and end timestamps.
    
    Args:
        video_path (Union[str, Path]): Path to the original video file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        output_path (Union[str, Path]): Path where the clip will be saved
        fade (bool, optional): Whether to add fade in/out effects. Defaults to True.
    
    Returns:
        str: Path to the created clip
    """
    try:
        with VideoFileClip(str(video_path)) as video:
            # Extract the subclip
            clip = video.subclip(start_time, end_time)
            
            # Add fade in/out effects if requested
            if fade and CLIP_FADE_DURATION > 0:
                clip = clip.fadein(CLIP_FADE_DURATION).fadeout(CLIP_FADE_DURATION)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the clip with specified settings
            clip.write_videofile(
                str(output_path),
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                bitrate=VIDEO_BITRATE,
                threads=4,
                logger=None  # Disable moviepy's console output
            )
            
            return str(output_path)
    except Exception as e:
        print(f"Error creating clip: {str(e)}")
        return None

def create_clips_from_dataframe(
    video_path: Union[str, Path],
    df: pd.DataFrame,
    output_dir: Union[str, Path] = None,
    video_id: str = None
) -> List[str]:
    """
    Create multiple video clips based on timestamps in a DataFrame.
    
    Args:
        video_path (Union[str, Path]): Path to the original video file
        df (pd.DataFrame): DataFrame with columns: start, end, text
        output_dir (Union[str, Path], optional): Directory to save clips. Defaults to CLIPS_OUTPUT_DIR.
        video_id (str, optional): Identifier for the video. Defaults to None.
    
    Returns:
        List[str]: List of paths to created clips
    """
    if output_dir is None:
        output_dir = CLIPS_OUTPUT_DIR
    
    if video_id is None:
        video_id = Path(video_path).stem
    
    # Validate DataFrame columns
    required_columns = {'start', 'end', 'text'}
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Create clips directory
    clips_dir = Path(output_dir) / video_id
    os.makedirs(clips_dir, exist_ok=True)
    
    created_clips = []
    
    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        # Create a filename using the start time and first few words of text
        text_prefix = "_".join(row['text'].split()[:5]).lower()
        text_prefix = "".join(c if c.isalnum() or c == '_' else '' for c in text_prefix)
        clip_name = f"{video_id}_{row['start']:.2f}_{text_prefix[:30]}.mp4"
        
        output_path = clips_dir / clip_name
        
        # Create the clip
        clip_path = create_clip(
            video_path=video_path,
            start_time=row['start'],
            end_time=row['end'],
            output_path=output_path
        )
        
        if clip_path:
            created_clips.append(clip_path)
    
    return created_clips

if __name__ == "__main__":
    pass