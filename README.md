# ClipScribe

## Overview

**ClipScribe** is a tool for extracting meaningful short clips from videos based on transcript analysis. It leverages OpenAI's Whisper model to transcribe video audio, segments the transcript into manageable chunks, and allows users to retrieve or generate short video clips based on specific instructions or queries. The transcript chunks are saved for future use, enabling efficient retrieval using Retrieval-Augmented Generation (RAG) techniques. The final output can be a summarized report of relevant information found in the script, a generated short film, or both.

## Project Structure

```
ClipScribe
├── src
│   ├── main.py        # Command line script for processing videos and generating clips
│   ├── app.py         # Gradio app for interactive use
│   └── utils.py       # Utility functions for audio extraction, transcription, and chunking
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ClipScribe
pip install -r requirements.txt
```

Or, to use a Conda environment with Python 3.11:

```bash
conda create -n ClipScribe python=3.11
conda activate ClipScribe
pip install -r requirements.txt
```

## Usage

### Command Line Interface

To process a video and extract relevant clips or summaries based on a user instruction, run:

```bash
python src/main.py <path_to_video_file> --instruction "<your_query_or_instruction>"
```

- This will transcribe the video, segment the transcript, and either generate a short film or summarize the relevant information found in the script according to your instruction.
- Transcript chunks are saved for future retrieval and RAG-based search.

### Gradio App

To launch the interactive Gradio app:

```bash
python src/app.py
```

- Upload a video, enter your instruction or query, and receive either a generated short clip or a summary of the relevant content.

## Features

- **Transcript Chunking:** Segments transcripts for efficient retrieval and RAG workflows.
- **Clip Generation:** Extracts and saves short video clips based on transcript relevance.
- **Summarization:** Summarizes information found in the script according to user instructions.
- **Persistent Storage:** Saves transcript chunks for future use and retrieval.

## Dependencies

- OpenAI Whisper (for transcription)
- Gradio (for web interface)
- MoviePy (for video/audio processing)
- NLTK (for sentence chunking)
- Other standard Python libraries

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
