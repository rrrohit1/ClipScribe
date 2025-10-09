# ClipScribe

**AI-powered video transcription and semantic search tool**

ClipScribe transcribes videos using Whisper AI, creates searchable transcript chunks with embeddings, and enables intelligent content discovery through semantic search and AI analysis.

## Features

- Whisper AI transcription with precise timestamps
- Intelligent sentence-based chunking with embeddings
- Semantic search using vector embeddings
- Gemini AI-powered content analysis
- Relevance-based ranking with similarity scores
- Persistent JSON storage for transcripts
- REST API and interactive web interface

## Quick Start

### Installation

```bash
git clone https://github.com/rrrohit1/ClipScribe.git
cd ClipScribe
conda env create -f environment.yaml
conda activate clipscribe
```

### Environment Setup

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
WHISPER_MODEL_SIZE=turbo
DEFAULT_CHUNK_SIZE=5
```

### Run the Application

```bash
python app.py
```

Access the Gradio interface at `http://localhost:7860`

## Usage Workflow

1. **Upload & Process**  
   - Upload your video
   - Choose semantic or standard chunking
   - Get transcript and chunk analysis

2. **Search & Analyze**  
   - Search by topic or keyword
   - Get AI-powered content analysis
   - View semantic relationships between chunks

3. **Create Clips**  
   - Extract clips by timestamp
   - Generate clips from search results
   - Automatic clip organization with text context

## Methodology

### Semantic Chunking Process

ClipScribe uses an intelligent approach to create meaningful chunks from video transcripts:

1. **Sliding Window Approach**
   - Uses a configurable window size (default: 4 sentences) with overlap
   - Slides through the transcript with specified overlap (default: 2 sentences)
   - Creates overlapping text segments for better context preservation

2. **Semantic Analysis**
   - Each chunk is embedded using Sentence Transformers (`all-MiniLM-L6-v2`)
   - Calculates cosine similarity between consecutive chunks
   - Creates a similarity score profile across the transcript

3. **Optimal Chunking**
   - Identifies local minima in similarity scores
   - These minima represent natural "semantic breaks" in content
   - Enforces constraints on chunk sizes:
     - Minimum distance: 2 sentences (prevents too-small chunks)
     - Maximum distance: 8 sentences (maintains digestible size)

4. **Advantages**
   - Creates context-aware chunks instead of fixed-size segments
   - Preserves semantic coherence within chunks
   - Enables more accurate search and retrieval
   - Optimizes for both human readability and AI processing

## Project Structure

```bash
ClipScribe/
├── app.py              # Gradio web interface
├── src/
│   ├── utils.py        # Core processing functions
│   ├── config.py       # Configuration & environment variables
│   ├── create_chunks.py # Semantic chunking implementation
│   └── create_video.py # Video clip generation
├── data/
│   ├── transcripts/    # Stored transcript chunks with embeddings
│   └── clips/          # Generated video clips
└── environment.yaml
```

## Interface Features

### Process Video

- Upload your video file
- Choose between standard or semantic chunking
- Get instant transcript with chunk analysis
- View similarity scores between chunks (with semantic chunking)

### Search Content

- Enter video ID from processing step
- Type your search query
- Adjust number of results (1-10)
- Get relevant segments with timestamps
- View similarity scores for each result

### AI Analysis

- Enter video ID and analysis instruction
- Get AI-powered insights about your content
- Receive suggested clips and timestamps
- Extract key information and summaries

### Extract Clips

- Create individual clips by timestamp
- Name your clips for easy reference
- Get instant preview of extracted segments
- Automatic fade effects (configurable)

## Dependencies

- **whisper** - Audio transcription
- **sentence-transformers** - Semantic embeddings
- **google-generativeai** - Gemini AI integration
- **moviepy** - Video/audio processing
- **flask** - REST API
- **gradio** - Web interface
- **scikit-learn** - Similarity calculations

## Configuration

Key settings in `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `turbo` | Whisper model (tiny/base/turbo) |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `DEFAULT_CHUNK_SIZE` | `5` | Sentences per chunk |
| `DEFAULT_TOP_K` | `5` | Search results to return |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum relevance score |

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contributing

Contributions welcome! Open an issue or submit a pull request.

---

**Current capabilities:** Video transcription, semantic search, AI-powered analysis
```