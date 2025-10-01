# Core Transcription & Storage
import os
import json
import tempfile
from pathlib import Path
import moviepy.editor as mp
import whisper
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Whisper model (using base model for balance of speed/accuracy)
# You can change to 'tiny', 'small', 'medium', or 'large' based on needs
WHISPER_MODEL = None

def get_whisper_model():
    """load Whisper model"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("turbo")
    return WHISPER_MODEL


def extract_audio(video_path, audio_path=None):
    """
    Extract audio from video file
    
    Args:
        video_path: Path to input video file
        audio_path: Path for output audio file (optional)
    
    Returns:
        Path to extracted audio file
    """
    if audio_path is None:
        # Create temporary audio file
        audio_path = tempfile.mktemp(suffix=".mp3")
    
    print(f"Extracting audio from {video_path}...")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    
    return audio_path


def transcribe_audio_with_whisper(audio_path):
    """
    Transcribe audio using OpenAI Whisper with timestamps
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        List of segments with text and timestamps
        [{
            'text': 'transcribed text',
            'start': start_time_seconds,
            'end': end_time_seconds
        }, ...]
    """
    print(f"Transcribing audio with Whisper...")
    model = get_whisper_model()
    
    result = model.transcribe(audio_path, verbose=False)
    
    # Extract segments with timestamps
    segments = []
    for segment in result['segments']:
        segments.append({
            'text': segment['text'].strip(),
            'start': segment['start'],
            'end': segment['end']
        })
    
    print(f"Transcription complete: {len(segments)} segments")
    return segments


def chunk_transcript_with_timestamps(segments, chunk_size=5):
    """
    Chunk transcript segments while preserving timing information
    
    Args:
        segments: List of transcript segments with timestamps
        chunk_size: Number of sentences per chunk
    
    Returns:
        List of chunks with text and time boundaries
        [{
            'text': 'chunked text',
            'start_time': start_seconds,
            'end_time': end_seconds,
            'sentence_count': number_of_sentences
        }, ...]
    """
    print(f"Chunking transcript into {chunk_size}-sentence segments...")
    
    # First, combine all text and tokenize into sentences
    full_text = " ".join([seg['text'] for seg in segments])
    sentences = nltk.sent_tokenize(full_text)
    
    chunks = []
    
    # Track current position in segments
    current_seg_idx = 0
    current_seg_char_pos = 0
    
    for i in range(0, len(sentences), chunk_size):
        chunk_sentences = sentences[i:i + chunk_size]
        chunk_text = ' '.join(chunk_sentences)
        
        # Find start time (timestamp of first character in chunk)
        chunk_start_char = len(' '.join(sentences[:i]))
        
        # Find end time (timestamp of last character in chunk)
        chunk_end_char = chunk_start_char + len(chunk_text)
        
        # Map character positions to segment timestamps
        start_time = None
        end_time = None
        
        char_count = 0
        for seg in segments:
            seg_len = len(seg['text'])
            
            # Find start time
            if start_time is None and char_count + seg_len >= chunk_start_char:
                start_time = seg['start']
            
            # Find end time
            if char_count + seg_len >= chunk_end_char:
                end_time = seg['end']
                break
            
            char_count += seg_len + 1  # +1 for space
        
        # Fallback if not found
        if start_time is None:
            start_time = segments[0]['start'] if segments else 0
        if end_time is None:
            end_time = segments[-1]['end'] if segments else 0
        
        chunks.append({
            'text': chunk_text,
            'start_time': start_time,
            'end_time': end_time,
            'sentence_count': len(chunk_sentences)
        })
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def save_transcript_chunks(video_id, chunks, storage_dir="data/transcripts"):
    """
    Save transcript chunks to disk for future RAG retrieval
    
    Args:
        video_id: Unique identifier for the video
        chunks: List of transcript chunks with metadata
        storage_dir: Directory to store transcript data
    """
    # Create storage directory if it doesn't exist
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    # Save chunks as JSON
    output_path = os.path.join(storage_dir, f"{video_id}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'chunks': chunks,
            'total_chunks': len(chunks)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")
    return output_path


def load_transcript_chunks(video_id, storage_dir="data/transcripts"):
    """
    Load previously saved transcript chunks
    
    Args:
        video_id: Unique identifier for the video
        storage_dir: Directory where transcript data is stored
    
    Returns:
        Dictionary with video_id and chunks, or None if not found
    """
    file_path = os.path.join(storage_dir, f"{video_id}.json")
    
    if not os.path.exists(file_path):
        print(f"No transcript found for video_id: {video_id}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {data['total_chunks']} chunks for {video_id}")
    return data