# Configuration with RAG settings
from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Whisper Configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "turbo")  # tiny, base, turbo
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu or cuda

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# Options: all-MiniLM-L6-v2 (fast, 384 dim), all-mpnet-base-v2 (better quality, 768 dim)

# RAG Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))  # Number of search results
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))  # Min similarity score

# Transcript Chunking
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "5"))  # sentences per chunk

# Storage Configuration
TRANSCRIPT_STORAGE_DIR = os.getenv("TRANSCRIPT_STORAGE_DIR", "data/transcripts")
CLIPS_OUTPUT_DIR = os.getenv("CLIPS_OUTPUT_DIR", "data/clips")

# Server Configuration
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

# Gemini Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

# Create necessary directories
os.makedirs(TRANSCRIPT_STORAGE_DIR, exist_ok=True)
os.makedirs(CLIPS_OUTPUT_DIR, exist_ok=True)

# Validation
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not set. AI analysis features will not work.")
    print("   Set it in your .env file: GEMINI_API_KEY=your_key_here")

print("✅ ClipScribe Phase 2 Configuration Loaded")
print(f"📁 Transcript Storage: {TRANSCRIPT_STORAGE_DIR}")
print(f"📁 Clips Output: {CLIPS_OUTPUT_DIR}")
print(f"🎙️  Whisper Model: {WHISPER_MODEL_SIZE}")
print(f"🔍 Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"🤖 Gemini Model: {GEMINI_MODEL}")
print(f"📊 Default Top-K Results: {DEFAULT_TOP_K}")
print(f"🎯 Similarity Threshold: {SIMILARITY_THRESHOLD}")