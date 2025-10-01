# RAG with Embeddings & Semantic Search
import os
import json
import tempfile
import numpy as np
from pathlib import Path
import moviepy.editor as mp
import whisper
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from src.config import GEMINI_API_KEY

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
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    print(f"Created embeddings for {len(chunks)} chunks")
    return chunks

def save_transcript_chunks(video_id, chunks, storage_dir="data/transcripts"):
    """Save transcript chunks with embeddings to disk"""
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(storage_dir, f"{video_id}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'chunks': chunks,
            'total_chunks': len(chunks)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks with embeddings to {output_path}")
    return output_path

def load_transcript_chunks(video_id, storage_dir="data/transcripts"):
    """Load previously saved transcript chunks with embeddings"""
    file_path = os.path.join(storage_dir, f"{video_id}.json")
    
    if not os.path.exists(file_path):
        print(f"No transcript found for video_id: {video_id}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {data['total_chunks']} chunks for {video_id}")
    return data

def semantic_search_chunks(query, chunks, top_k=5):
    """
    Perform semantic search on transcript chunks using RAG
    
    Args:
        query: Search query string
        chunks: List of chunks with embeddings
        top_k: Number of top results to return
    
    Returns:
        List of most relevant chunks with similarity scores
    """
    print(f"Performing semantic search for: '{query}'")
    model = get_embedding_model()
    
    # Create embedding for query
    query_embedding = model.encode([query])[0]
    
    # Extract embeddings from chunks
    chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])
    
    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Prepare results
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['similarity'] = float(similarities[idx])
        # Remove embedding from result (too large)
        chunk.pop('embedding', None)
        results.append(chunk)
    
    print(f"Found {len(results)} relevant chunks")
    return results

def analyze_with_gemini(full_transcript, instruction, chunks=None):
    """
    Analyze video content using Gemini AI
    
    Args:
        full_transcript: Complete transcript text
        instruction: User's analysis instruction
        chunks: Optional list of chunks with timestamps
    
    Returns:
        AI-generated analysis
    """
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
    """
    Find all moments in a video relevant to a query above a similarity threshold
    
    Args:
        query: Search query
        video_id: Video identifier
        threshold: Minimum similarity score (0-1)
    
    Returns:
        List of relevant moments with timestamps
    """
    data = load_transcript_chunks(video_id)
    if not data:
        return []
    
    # Get all chunks above threshold
    all_results = semantic_search_chunks(query, data['chunks'], top_k=len(data['chunks']))
    relevant_moments = [r for r in all_results if r['similarity'] >= threshold]
    
    print(f"Found {len(relevant_moments)} moments above threshold {threshold}")
    return relevant_moments