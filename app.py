import gradio as gr
from src.utils import (
    extract_audio, transcribe_audio_with_whisper, 
    chunk_transcript_with_timestamps, save_transcript_chunks,
    load_transcript_chunks, create_embeddings_for_chunks,
    semantic_search_chunks, analyze_with_gemini,
    extract_clip_from_video, compile_clips, get_video_path
)
import os
import tempfile
import json
from pathlib import Path

def process_video(video_file):
    """Process video: extract audio, transcribe, chunk, embed, and store"""
    if video_file is None:
        return {"error": "No video file provided"}
    
    # Get the video path
    video_path = str(Path(video_file.name))
    
    # Extract audio and transcribe
    audio_file = extract_audio(video_path)
    transcript_segments = transcribe_audio_with_whisper(audio_file)
    chunks = chunk_transcript_with_timestamps(transcript_segments, chunk_size=5)
    chunks_with_embeddings = create_embeddings_for_chunks(chunks)
    
    # Generate video ID and save chunks
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    save_transcript_chunks(video_id, chunks_with_embeddings, video_path=video_path)
    
    full_transcript = " ".join([seg['text'] for seg in transcript_segments])
    
    return {
        "video_id": video_id,
        "full_transcript": full_transcript,
        "chunks": chunks_with_embeddings,
        "total_chunks": len(chunks_with_embeddings)
    }

def search_video_content(video_id, query, top_k=5):
    """Search for relevant content in a video using semantic search"""
    data = load_transcript_chunks(video_id)
    if not data:
        return {"error": f"Video {video_id} not found"}
    
    results = semantic_search_chunks(query, data['chunks'], top_k=top_k)
    
    return {
        "video_id": video_id,
        "query": query,
        "results": results,
        "total_results": len(results)
    }

def analyze_video_content(video_id, instruction):
    """Analyze video content using Gemini AI based on user instruction"""
    data = load_transcript_chunks(video_id)
    if not data:
        return {"error": f"Video {video_id} not found"}
    
    full_transcript = " ".join([chunk['text'] for chunk in data['chunks']])
    analysis = analyze_with_gemini(full_transcript, instruction, data['chunks'])
    
    return {
        "video_id": video_id,
        "instruction": instruction,
        "analysis": analysis
    }

def generate_clip(video_id, start_time, end_time, clip_name=None):
    """Extract a single clip from video based on timestamps"""
    video_path = get_video_path(video_id)
    if not video_path:
        return {"error": f"Video file for {video_id} not found"}
    
    clip_path = extract_clip_from_video(video_path, start_time, end_time, video_id, clip_name)
    
    return {
        "video_id": video_id,
        "clip_path": clip_path,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time
    }

def generate_compilation(video_id, search_results, output_name=None):
    """Compile multiple search results into a single video"""
    video_path = get_video_path(video_id)
    if not video_path:
        return {"error": f"Video file for {video_id} not found"}
    
    # Extract timestamps from search results
    clips_info = [(r['start_time'], r['end_time']) for r in search_results]
    
    compilation_path = compile_clips(video_path, clips_info, video_id, output_name)
    
    return {
        "video_id": video_id,
        "compilation_path": compilation_path,
        "total_clips": len(clips_info),
        "clips": clips_info
    }

# Gradio interfaces
def gradio_process_video(video_file):
    """Gradio interface for video processing"""
    if video_file is None:
        return "Please upload a video file."
    
    result = process_video(video_file.name)
    
    output = f"""
**Video ID:** {result['video_id']}

**Processing Summary:**
- Total chunks created: {result['total_chunks']}
- Chunks embedded and saved for RAG retrieval

**Full Transcript:**
{result['full_transcript'][:1000]}...

✅ Video processed successfully! Use the Search or Analyze tabs to explore content.
"""
    return output

def gradio_search_video(video_id, query, top_k):
    """Gradio interface for semantic search"""
    if not video_id or not query:
        return "Please provide both Video ID and search query.", None
    
    result = search_video_content(video_id, query, int(top_k))
    
    if "error" in result:
        return f"❌ Error: {result['error']}", None
    
    output = f"""
**Search Results for:** "{query}"
**Video ID:** {video_id}
**Found {result['total_results']} relevant segments:**

"""
    for i, res in enumerate(result['results'], 1):
        output += f"""
---
**Result {i}** (Similarity: {res['similarity']:.3f})
⏱️  Time: {res['start_time']:.2f}s - {res['end_time']:.2f}s
📝 Text: {res['text']}

"""
    
    # Return results as JSON for clip generation
    results_json = json.dumps(result['results'], indent=2)
    return output, results_json

def gradio_analyze_video(video_id, instruction):
    """Gradio interface for AI analysis"""
    if not video_id or not instruction:
        return "Please provide both Video ID and analysis instruction."
    
    result = analyze_video_content(video_id, instruction)
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    output = f"""
**AI Analysis**
**Video ID:** {video_id}
**Instruction:** {instruction}

---

{result['analysis']}
"""
    return output

def gradio_extract_clip(video_id, start_time, end_time, clip_name):
    """Gradio interface for single clip extraction"""
    if not video_id or start_time is None or end_time is None:
        return "Please provide Video ID, start time, and end time.", None
    
    try:
        start = float(start_time)
        end = float(end_time)
        
        result = generate_clip(video_id, start, end, clip_name if clip_name else None)
        
        if "error" in result:
            return f"❌ Error: {result['error']}", None
        
        output = f"""
✅ **Clip extracted successfully!**

**Video ID:** {result['video_id']}
**Duration:** {result['duration']:.2f} seconds
**Clip saved to:** {result['clip_path']}

You can download the clip using the file path above.
"""
        return output, result['clip_path']
    except ValueError:
        return "Invalid time values. Please enter numbers.", None

def gradio_compile_clips(video_id, search_results_json, output_name):
    """Gradio interface for compiling multiple clips"""
    if not video_id or not search_results_json:
        return "Please provide Video ID and search results.", None
    
    try:
        search_results = json.loads(search_results_json)
        
        result = generate_compilation(video_id, search_results, output_name if output_name else None)
        
        if "error" in result:
            return f"❌ Error: {result['error']}", None
        
        output = f"""
✅ **Compilation created successfully!**

**Video ID:** {result['video_id']}
**Total clips compiled:** {result['total_clips']}
**Compilation saved to:** {result['compilation_path']}

**Clips included:**
"""
        for i, (start, end) in enumerate(result['clips'], 1):
            output += f"\n{i}. {start:.2f}s - {end:.2f}s ({end-start:.2f}s)"
        
        return output, result['compilation_path']
    except json.JSONDecodeError:
        return "Invalid search results format.", None

# Create Gradio interface with tabs
with gr.Blocks(title="ClipScribe - Phase 3") as iface:
    gr.Markdown("# ClipScribe - Video Transcription, Search & Clip Generation")
    gr.Markdown("Process videos, search content semantically, analyze with AI, and extract clips")
    
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
        search_results_json = gr.Textbox(label="Search Results (JSON)", lines=5, visible=False)
        search_btn.click(gradio_search_video, 
                        inputs=[search_video_id, search_query, search_top_k], 
                        outputs=[search_output, search_results_json])
    
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
    
    with gr.Tab("✂️ Extract Single Clip"):
        with gr.Row():
            clip_video_id = gr.Textbox(label="Video ID", placeholder="Enter video ID")
        with gr.Row():
            clip_start = gr.Number(label="Start Time (seconds)", value=0)
            clip_end = gr.Number(label="End Time (seconds)", value=10)
        clip_name_input = gr.Textbox(label="Clip Name (optional)", placeholder="my_clip")
        extract_btn = gr.Button("Extract Clip", variant="primary")
        extract_output = gr.Textbox(label="Extraction Result", lines=10)
        clip_path_output = gr.Textbox(label="Clip Path", visible=False)
        extract_btn.click(gradio_extract_clip,
                         inputs=[clip_video_id, clip_start, clip_end, clip_name_input],
                         outputs=[extract_output, clip_path_output])
    
    with gr.Tab("🎬 Compile Clips"):
        gr.Markdown("**Note:** First perform a search, then use the results JSON here")
        with gr.Row():
            compile_video_id = gr.Textbox(label="Video ID", placeholder="Enter video ID")
        compile_results = gr.Textbox(
            label="Search Results JSON", 
            placeholder="Paste search results JSON from Search tab",
            lines=5
        )
        compile_name = gr.Textbox(label="Compilation Name (optional)", placeholder="my_compilation")
        compile_btn = gr.Button("Create Compilation", variant="primary")
        compile_output = gr.Textbox(label="Compilation Result", lines=15)
        compilation_path_output = gr.Textbox(label="Compilation Path", visible=False)
        compile_btn.click(gradio_compile_clips,
                         inputs=[compile_video_id, compile_results, compile_name],
                         outputs=[compile_output, compilation_path_output])

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )