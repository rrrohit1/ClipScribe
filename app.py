# Core Transcription & Storage
from flask import Flask, request, jsonify
import gradio as gr
from src.utils import extract_audio, transcribe_audio_with_whisper, chunk_transcript_with_timestamps, save_transcript_chunks
import os
import tempfile

app = Flask(__name__)

def process_video(video_file):
    """Process video: extract audio, transcribe, chunk, and store"""
    # Extract audio from video
    audio_file = extract_audio(video_file)
    
    # Transcribe with Whisper (returns segments with timestamps)
    transcript_segments = transcribe_audio_with_whisper(audio_file)
    
    # Chunk transcript while preserving timestamps
    chunks = chunk_transcript_with_timestamps(transcript_segments, chunk_size=5)
    
    # Save chunks for future RAG retrieval
    video_id = os.path.splitext(os.path.basename(video_file))[0]
    save_transcript_chunks(video_id, chunks)
    
    # Return full transcript and chunk info
    full_transcript = " ".join([seg['text'] for seg in transcript_segments])
    
    return {
        "video_id": video_id,
        "full_transcript": full_transcript,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """API endpoint for video transcription"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        result = process_video(tmp_path)
        return jsonify(result)
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Gradio interface
def gradio_interface(video_file):
    """Gradio interface for video processing"""
    if video_file is None:
        return "Please upload a video file."
    
    result = process_video(video_file.name)
    
    # Format output for display
    output = f"""
    Video ID: {result['video_id']}

    Full Transcript:
    {result['full_transcript']}

    Processing Summary:
    - Total chunks created: {result['total_chunks']}
    - Chunks saved for RAG retrieval

    Sample Chunks (first 3):
    """
    for i, chunk in enumerate(result['chunks'][:3], 1):
        output += f"\nChunk {i} [{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s]:\n{chunk['text'][:200]}...\n"
    
    return output

iface = gr.Interface(
    fn=gradio_interface, 
    inputs=gr.Video(label="Upload Video"), 
    outputs=gr.Textbox(label="Transcription Result", lines=20),
    title="ClipScribe - Video Transcription", 
    description="Upload a video file to transcribe its audio using Whisper AI. Transcripts are chunked and stored for future clip extraction."
)

if __name__ == "__main__":
    # Run both Flask and Gradio
    # Note: In production, run these separately
    import threading
    
    # Start Flask in a thread
    flask_thread = threading.Thread(target=lambda: app.run(debug=False, port=5000))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Launch Gradio
    iface.launch(server_port=7860)