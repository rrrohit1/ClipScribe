from flask import Flask, request, jsonify
import gradio as gr
from utils import extract_audio, transcribe_audio

app = Flask(__name__)

def transcribe_video(video_file):
    audio_file = extract_audio(video_file)
    transcription = transcribe_audio(audio_file)
    return transcription

@app.route("/transcribe", methods=["POST"])
def transcribe():
    video_file = request.files['video']
    transcription = transcribe_video(video_file)
    return jsonify({"transcription": transcription})

if __name__ == "__main__":
    app.run(debug=True)

# Gradio interface
def gradio_interface(video_file):
    transcription = transcribe_video(video_file.name)
    return transcription

iface = gr.Interface(fn=gradio_interface, inputs="file", outputs="text", title="Video Transcription App", description="Upload a video file to transcribe its audio.")
iface.launch()