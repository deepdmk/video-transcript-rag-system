# YouTube Transcript Q&A

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Gradio](https://img.shields.io/badge/Gradio-5.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A RAG-based application that extracts YouTube video transcripts and enables video summarization and question-answering through a Gradio web interface, powered by IBM WatsonX and FAISS vector search.

## Problem Statement

Watching lengthy YouTube videos to find specific information is time-consuming. This project enables users to quickly summarize videos and ask natural language questions about video content without watching the entire video.

## Features

- Automatic transcript extraction from YouTube videos
- Video summarization using LLM-generated summaries
- RAG-based Q&A with semantic search over transcript chunks
- Interactive Gradio web interface

## Quick Start

### Prerequisites

```bash
pip install gradio youtube-transcript-api langchain langchain-ibm langchain-community ibm-watsonx-ai faiss-cpu
```

### Configuration

Update `setup_credentials()` with your IBM WatsonX credentials and project ID.

### Run

```bash
python ytbot.py
```

Navigate to `http://localhost:7860` in your browser.

## Architecture

| Component | Details |
|-----------|---------|
| **LLM** | Llama 3.2 3B Instruct (via WatsonX) |
| **Embeddings** | IBM SLATE-30M-ENG |
| **Vector Store** | FAISS |
| **Text Splitter** | RecursiveCharacterTextSplitter (200 chunk / 20 overlap) |
| **Decoding** | Greedy (max 900 tokens) |
| **Interface** | Gradio Blocks |

## Project Structure

```
├── ytbot.py
└── README.md
```

## How It Works

1. **Transcript Extraction** — Fetches English transcripts via YouTube Transcript API (prefers manual over auto-generated)
2. **Summarization** — Sends full transcript to LLM with summarization prompt
3. **Q&A Pipeline** — Chunks transcript → embeds with SLATE → indexes in FAISS → retrieves relevant chunks → generates answer

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG) architecture
- LangChain prompt templates and chains
- Vector similarity search with FAISS
- API integration (YouTube, IBM WatsonX)
- Web application development with Gradio

## License

MIT

## Acknowledgments

- Project completed as part of IBM Advanced RAG & Agentic AI Certification
- Models: Llama 3.2 (Meta), SLATE-30M (IBM)
