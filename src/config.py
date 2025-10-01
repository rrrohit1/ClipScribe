from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # For future use with embeddings

# Whisper Configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "turbo")  # tiny, base, turbo
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu or cuda

# Transcript Chunking
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "5"))  # sentences per chunk

# Storage Configuration
TRANSCRIPT_STORAGE_DIR = os.getenv("TRANSCRIPT_STORAGE_DIR", "data/transcripts")
CLIPS_OUTPUT_DIR = os.getenv("CLIPS_OUTPUT_DIR", "data/clips")

# Server Configuration
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

# Create necessary directories
os.makedirs(TRANSCRIPT_STORAGE_DIR, exist_ok=True)
os.makedirs(CLIPS_OUTPUT_DIR, exist_ok=True)

print("ClipScribe Configuration Loaded")
print(f"Transcript Storage: {TRANSCRIPT_STORAGE_DIR}")
print(f"Clips Output: {CLIPS_OUTPUT_DIR}")
print(f"Whisper Model: {WHISPER_MODEL_SIZE}")