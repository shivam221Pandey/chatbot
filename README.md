# Local AI RAG Agent

A powerful, local AI agent capable of ingesting PDFs, answering questions with citations, and generating quizzes.

## Features
- **Knowledge Base**: Upload multiple PDFs to train the agent.
- **Precision**: Answers questions with exact page number citations.
- **Quiz Generator**: Create MCQs or Descriptive questions from your documents.
- **Flexible AI**: Supports cloud models (Gemini) or fully local models (Ollama).

## Setup

1. **Install Python**: Ensure Python 3.9+ is installed and added to your PATH.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Configuration

- **Gemini (Cloud)**: Get a free API key from [Google AI Studio](https://aistudio.google.com/).
- **Ollama (Local)**: Install [Ollama](https://ollama.com/) and pull a model (e.g., `ollama pull llama3`).

## Project Structure
- `app.py`: The Main User Interface.
- `rag_engine.py`: The backend logic for indexing and querying.
- `requirements.txt`: List of dependencies.
- `data/`: Directory where uploaded PDFs are stored.
- `storage/`: Directory where the vector index is persisted.
