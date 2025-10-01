import sys
import os
import whisper
from utils import extract_audio, chunk_sentences

def main(video_file_path):
    if not os.path.isfile(video_file_path):
        print(f"Error: The file {video_file_path} does not exist.")
        return

    # Extract audio from the video file
    audio_file_path = extract_audio(video_file_path)

    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    transcription = model.transcribe(audio_file_path)["text"]

    # Chunk the transcription into 5 sentence parts
    chunks = chunk_sentences(transcription, 5)

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_file_path>")
    else:
        main(sys.argv[1])
