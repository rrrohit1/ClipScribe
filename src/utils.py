import os
import json
import tempfile
import numpy as np
from pathlib import Path
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from src.config import GEMINI_API_KEY, CLIPS_OUTPUT_DIR, TRANSCRIPT_STORAGE_DIR
import shutil

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize models
WHISPER_MODEL = None
EMBEDDING_MODEL = None

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def get_whisper_model():
    """Load Whisper model"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("turbo")
    return WHISPER_MODEL

def get_embedding_model():
    """Load embedding model"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return EMBEDDING_MODEL

def extract_audio(video_path, audio_path=None):
    """Extract audio from video file"""
    if audio_path is None:
        audio_path = tempfile.mktemp(suffix=".mp3")
    
    print(f"Extracting audio from {video_path}...")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    
    return audio_path

def transcribe_audio_with_whisper(audio_path):
    """Transcribe audio using OpenAI Whisper with timestamps"""
    print(f"Transcribing audio with Whisper...")
    model = get_whisper_model()
    
    result = model.transcribe(audio_path, verbose=False)
    
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
    """Chunk transcript segments while preserving timing information"""
    print(f"Chunking transcript into {chunk_size}-sentence segments...")
    
    full_text = " ".join([seg['text'] for seg in segments])
    sentences = nltk.sent_tokenize(full_text)
    
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk_sentences = sentences[i:i + chunk_size]
        chunk_text = ' '.join(chunk_sentences)
        
        chunk_start_char = len(' '.join(sentences[:i]))
        chunk_end_char = chunk_start_char + len(chunk_text)
        
        start_time = None
        end_time = None
        
        char_count = 0
        for seg in segments:
            seg_len = len(seg['text'])
            
            if start_time is None and char_count + seg_len >= chunk_start_char:
                start_time = seg['start']
            
            if char_count + seg_len >= chunk_end_char:
                end_time = seg['end']
                break
            
            char_count += seg_len + 1
        
        if start_time is None:
            start_time = segments[0]['start'] if segments else 0
        if end_time is None:
            end_time = segments[-1]['end'] if segments else 0
        
        chunks.append({
            'text': chunk_text,
            'start_time': start_time,
            'end_time': end_time,
            'sentence_count': len(chunk_sentences),
            'chunk_id': i // chunk_size
        })
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_embeddings_for_chunks(chunks):
    """Create vector embeddings for each chunk for semantic search"""
    print("Creating embeddings for chunks...")
    model = get_embedding_model()
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    print(f"Created embeddings for {len(chunks)} chunks")
    return chunks

def save_transcript_chunks(video_id, chunks, storage_dir=None, video_path=None):
    """Save transcript chunks with embeddings and video path reference"""
    if storage_dir is None:
        storage_dir = TRANSCRIPT_STORAGE_DIR
    
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy video to data directory for future clip extraction
    if video_path and os.path.exists(video_path):
        video_storage_dir = os.path.join("data", "videos")
        Path(video_storage_dir).mkdir(parents=True, exist_ok=True)
        
        video_ext = os.path.splitext(video_path)[1]
        stored_video_path = os.path.join(video_storage_dir, f"{video_id}{video_ext}")
        
        if not os.path.exists(stored_video_path):
            shutil.copy2(video_path, stored_video_path)
            print(f"Copied video to {stored_video_path}")
    
    output_path = os.path.join(storage_dir, f"{video_id}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'chunks': chunks,
            'total_chunks': len(chunks),
            'video_path': stored_video_path if video_path else None
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks with embeddings to {output_path}")
    return output_path

def load_transcript_chunks(video_id, storage_dir=None):
    """Load previously saved transcript chunks with embeddings"""
    if storage_dir is None:
        storage_dir = TRANSCRIPT_STORAGE_DIR
    
    file_path = os.path.join(storage_dir, f"{video_id}.json")
    
    if not os.path.exists(file_path):
        print(f"No transcript found for video_id: {video_id}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {data['total_chunks']} chunks for {video_id}")
    return data

def semantic_search_chunks(query, chunks, top_k=5):
    """Perform semantic search on transcript chunks using RAG"""
    print(f"Performing semantic search for: '{query}'")
    model = get_embedding_model()
    
    query_embedding = model.encode([query])[0]
    chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['similarity'] = float(similarities[idx])
        chunk.pop('embedding', None)
        results.append(chunk)
    
    print(f"Found {len(results)} relevant chunks")
    return results

def analyze_with_gemini(full_transcript, instruction, chunks=None):
    """Analyze video content using Gemini AI"""
    print(f"Analyzing with Gemini: '{instruction}'")
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are analyzing a video transcript. The user wants you to: {instruction}

Full Transcript:
{full_transcript}

Please provide a detailed analysis based on the instruction. If relevant, reference specific timestamps or sections of the content.
"""
        
        response = model.generate_content(prompt)
        analysis = response.text
        
        print("Analysis complete")
        return analysis
        
    except Exception as e:
        error_msg = f"Error during Gemini analysis: {str(e)}"
        print(error_msg)
        return error_msg

def find_relevant_moments(query, video_id, threshold=0.5):
    """Find all moments in a video relevant to a query above a similarity threshold"""
    data = load_transcript_chunks(video_id)
    if not data:
        return []
    
    all_results = semantic_search_chunks(query, data['chunks'], top_k=len(data['chunks']))
    relevant_moments = [r for r in all_results if r['similarity'] >= threshold]
    
    print(f"Found {len(relevant_moments)} moments above threshold {threshold}")
    return relevant_moments

def get_video_path(video_id):
    """Get the stored video path for a given video_id"""
    data = load_transcript_chunks(video_id)
    if not data:
        return None
    
    video_path = data.get('video_path')
    
    if video_path and os.path.exists(video_path):
        return video_path
    
    # Fallback: search in videos directory
    video_dir = os.path.join("data", "videos")
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            if file.startswith(video_id):
                return os.path.join(video_dir, file)
    
    print(f"Video file not found for {video_id}")
    return None

def extract_clip_from_video(video_path, start_time, end_time, video_id, clip_name=None):
    """
    Extract a clip from video based on start and end timestamps
    
    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        video_id: Video identifier
        clip_name: Optional custom name for the clip
    
    Returns:
        Path to the extracted clip
    """
    print(f"Extracting clip from {start_time}s to {end_time}s...")
    
    Path(CLIPS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate clip filename
    if clip_name:
        clip_filename = f"{video_id}_{clip_name}.mp4"
    else:
        clip_filename = f"{video_id}_clip_{start_time:.0f}_{end_time:.0f}.mp4"
    
    clip_path = os.path.join(CLIPS_OUTPUT_DIR, clip_filename)
    
    # Load video and extract clip
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)
    
    # Write clip to file
    clip.write_videofile(clip_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    
    # Clean up
    clip.close()
    video.close()
    
    print(f"Clip saved to {clip_path}")
    return clip_path

def compile_clips(video_path, clips_info, video_id, output_name=None):
    """
    Compile multiple clips into a single video
    
    Args:
        video_path: Path to source video
        clips_info: List of (start_time, end_time) tuples
        video_id: Video identifier
        output_name: Optional custom name for compilation
    
    Returns:
        Path to the compiled video
    """
    print(f"Compiling {len(clips_info)} clips into single video...")
    
    Path(CLIPS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate compilation filename
    if output_name:
        compilation_filename = f"{video_id}_{output_name}_compilation.mp4"
    else:
        compilation_filename = f"{video_id}_compilation_{len(clips_info)}_clips.mp4"
    
    compilation_path = os.path.join(CLIPS_OUTPUT_DIR, compilation_filename)
    
    # Load video
    video = VideoFileClip(video_path)
    
    # Extract all clips
    clips = []
    for start_time, end_time in clips_info:
        clip = video.subclip(start_time, end_time)
        clips.append(clip)
    
    # Concatenate clips
    final_clip = mp.concatenate_videoclips(clips, method="compose")
    
    # Write compilation to file
    final_clip.write_videofile(compilation_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    
    # Clean up
    for clip in clips:
        clip.close()
    final_clip.close()
    video.close()
    
    print(f"Compilation saved to {compilation_path}")
    return compilation_path

def extract_clips_from_search(video_id, query, top_k=5, compile_output=True):
    """
    High-level function: search for content and extract clips
    
    Args:
        video_id: Video identifier
        query: Search query
        top_k: Number of clips to extract
        compile_output: Whether to compile clips into single video
    
    Returns:
        Dictionary with clip paths and metadata
    """
    # Search for relevant content
    data = load_transcript_chunks(video_id)
    if not data:
        return {"error": f"Video {video_id} not found"}
    
    results = semantic_search_chunks(query, data['chunks'], top_k=top_k)
    
    # Get video path
    video_path = get_video_path(video_id)
    if not video_path:
        return {"error": f"Video file for {video_id} not found"}
    
    # Extract individual clips
    clip_paths = []
    clips_info = []
    
    for i, result in enumerate(results, 1):
        clip_path = extract_clip_from_video(
            video_path, 
            result['start_time'], 
            result['end_time'],
            video_id,
            clip_name=f"search_{i}"
        )
        clip_paths.append(clip_path)
        clips_info.append((result['start_time'], result['end_time']))
    
    # Optionally compile all clips
    compilation_path = None
    if compile_output and len(clips_info) > 1:
        compilation_path = compile_clips(video_path, clips_info, video_id, f"search_{query[:20]}")
    
    return {
        "video_id": video_id,
        "query": query,
        "individual_clips": clip_paths,
        "compilation": compilation_path,
        "clips_info": clips_info,
        "search_results": results
    }