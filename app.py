# RAG-based Semantic Search & Analysis
from flask import Flask, request, jsonify
import gradio as gr
from src.utils import (
    extract_audio, transcribe_audio_with_whisper, 
    chunk_transcript_with_timestamps, save_transcript_chunks,
    load_transcript_chunks, create_embeddings_for_chunks,
    semantic_search_chunks, analyze_with_gemini
)
import os
import tempfile

app = Flask(__name__)

def process_video(video_file):
    """Process video: extract audio, transcribe, chunk, embed, and store"""
    # Extract audio from video
    audio_file = extract_audio(video_file)
    
    # Transcribe with Whisper (returns segments with timestamps)
    transcript_segments = transcribe_audio_with_whisper(audio_file)
    
    # Chunk transcript while preserving timestamps
    chunks = chunk_transcript_with_timestamps(transcript_segments, chunk_size=5)
    
    # Create embeddings for semantic search
    chunks_with_embeddings = create_embeddings_for_chunks(chunks)
    
    # Save chunks with embeddings for future RAG retrieval
    video_id = os.path.splitext(os.path.basename(video_file))[0]
    save_transcript_chunks(video_id, chunks_with_embeddings)
    
    # Return full transcript and chunk info
    full_transcript = " ".join([seg['text'] for seg in transcript_segments])
    
    return {
        "video_id": video_id,
        "full_transcript": full_transcript,
        "chunks": chunks_with_embeddings,
        "total_chunks": len(chunks_with_embeddings)
    }

def search_video_content(video_id, query, top_k=5):
    """Search for relevant content in a video using semantic search"""
    # Load transcript chunks
    data = load_transcript_chunks(video_id)
    if not data:
        return {"error": f"Video {video_id} not found"}
    
    # Perform semantic search
    results = semantic_search_chunks(query, data['chunks'], top_k=top_k)
    
    return {
        "video_id": video_id,
        "query": query,
        "results": results,
        "total_results": len(results)
    }

def analyze_video_content(video_id, instruction):
    """Analyze video content using Gemini AI based on user instruction"""
    # Load transcript chunks
    data = load_transcript_chunks(video_id)
    if not data:
        return {"error": f"Video {video_id} not found"}
    
    # Get full transcript
    full_transcript = " ".join([chunk['text'] for chunk in data['chunks']])
    
    # Analyze with Gemini
    analysis = analyze_with_gemini(full_transcript, instruction, data['chunks'])
    
    return {
        "video_id": video_id,
        "instruction": instruction,
        "analysis": analysis
    }

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """API endpoint for video transcription and processing"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        result = process_video(tmp_path)
        # Remove embeddings from response (too large)
        result['chunks'] = [{k: v for k, v in chunk.items() if k != 'embedding'} 
                           for chunk in result['chunks']]
        return jsonify(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route("/search", methods=["POST"])
def search():
    """API endpoint for semantic search in video transcripts"""
    data = request.json
    video_id = data.get('video_id')
    query = data.get('query')
    top_k = data.get('top_k', 5)
    
    if not video_id or not query:
        return jsonify({"error": "video_id and query are required"}), 400
    
    result = search_video_content(video_id, query, top_k)
    return jsonify(result)

@app.route("/analyze", methods=["POST"])
def analyze():
    """API endpoint for AI-powered video content analysis"""
    data = request.json
    video_id = data.get('video_id')
    instruction = data.get('instruction')
    
    if not video_id or not instruction:
        return jsonify({"error": "video_id and instruction are required"}), 400
    
    result = analyze_video_content(video_id, instruction)
    return jsonify(result)

# Gradio interfaces
def gradio_process_video(video_file):
    """Gradio interface for video processing"""
    if video_file is None:
        return "Please upload a video file."
    
    result = process_video(video_file.name)
    
    output = f"""
📹 **Video ID:** {result['video_id']}

📊 **Processing Summary:**
- Total chunks created: {result['total_chunks']}
- Chunks embedded and saved for RAG retrieval

📝 **Full Transcript:**
{result['full_transcript'][:1000]}...

✅ Video processed successfully! Use the Search or Analyze tabs to explore content.
"""
    return output

def gradio_search_video(video_id, query, top_k):
    """Gradio interface for semantic search"""
    if not video_id or not query:
        return "Please provide both Video ID and search query."
    
    result = search_video_content(video_id, query, int(top_k))
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    output = f"""
🔍 **Search Results for:** "{query}"
📹 **Video ID:** {video_id}
📊 **Found {result['total_results']} relevant segments:**

"""
    for i, res in enumerate(result['results'], 1):
        output += f"""
---
**Result {i}** (Similarity: {res['similarity']:.3f})
⏱️  Time: {res['start_time']:.2f}s - {res['end_time']:.2f}s
📝 Text: {res['text']}

"""
    return output

def gradio_analyze_video(video_id, instruction):
    """Gradio interface for AI analysis"""
    if not video_id or not instruction:
        return "Please provide both Video ID and analysis instruction."
    
    result = analyze_video_content(video_id, instruction)
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    output = f"""
🤖 **AI Analysis**
📹 **Video ID:** {video_id}
📋 **Instruction:** {instruction}

---

{result['analysis']}
"""
    return output

# Create Gradio interface with tabs
with gr.Blocks(title="🎬 ClipScribe - Phase 2") as iface:
    gr.Markdown("# 🎬 ClipScribe - Video Transcription & Intelligent Search")
    gr.Markdown("Process videos, search content semantically, and analyze with AI")
    
    with gr.Tab("📤 Process Video"):
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
        process_btn = gr.Button("Process Video", variant="primary")
        process_output = gr.Textbox(label="Processing Result", lines=15)
        process_btn.click(gradio_process_video, inputs=video_input, outputs=process_output)
    
    with gr.Tab("🔍 Search Content"):
        with gr.Row():
            search_video_id = gr.Textbox(label="Video ID", placeholder="Enter video ID from processing")
            search_query = gr.Textbox(label="Search Query", placeholder="What are you looking for?")
            search_top_k = gr.Slider(1, 10, value=5, step=1, label="Number of Results")
        search_btn = gr.Button("Search", variant="primary")
        search_output = gr.Textbox(label="Search Results", lines=15)
        search_btn.click(gradio_search_video, 
                        inputs=[search_video_id, search_query, search_top_k], 
                        outputs=search_output)
    
    with gr.Tab("🤖 AI Analysis"):
        with gr.Row():
            analyze_video_id = gr.Textbox(label="Video ID", placeholder="Enter video ID from processing")
        analyze_instruction = gr.Textbox(
            label="Analysis Instruction", 
            placeholder="E.g., 'Summarize the key points' or 'Find moments discussing X'",
            lines=3
        )
        analyze_btn = gr.Button("Analyze", variant="primary")
        analyze_output = gr.Textbox(label="AI Analysis", lines=15)
        analyze_btn.click(gradio_analyze_video, 
                         inputs=[analyze_video_id, analyze_instruction], 
                         outputs=analyze_output)

if __name__ == "__main__":
    import threading
    
    # Start Flask in a thread
    flask_thread = threading.Thread(target=lambda: app.run(debug=False, port=5000))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Launch Gradio
    iface.launch(server_port=7860)