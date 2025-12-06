# AI Tutor Multi-Agent System

A FastAPI backend API for an AI-powered tutoring system that helps students learn from PDF documents.

## Features

- **PDF Summarization** - Upload a PDF and get an AI-generated summary
- **Quiz Generation** - Create quizzes based on document content with optional focus areas
- **Socratic Tutoring** - Chat with an AI tutor that guides learning through questions rather than direct answers

## Tech Stack

- FastAPI
- LangChain + LangGraph
- Google Gemini (gemini-2.0-flash-lite)
- Sentence Transformers for embeddings

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/summarizer` | POST | Upload PDF, get summary and session ID |
| `/examiner` | POST | Generate quiz from session (optional focus comment) |
| `/supervisor` | POST | Chat with Socratic tutor, get feedback on answers |
| `/session/{id}` | GET | Get session info |
| `/session/{id}` | DELETE | Delete session |
| `/health` | GET | Health check |

## Setup

1. Set `GOOGLE_API_KEY` environment variable
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `uvicorn app:app --host 0.0.0.0 --port 8000`

## Usage Flow

1. Upload PDF to `/summarizer` → receive `session_id`
2. Generate quiz via `/examiner` with `session_id`
3. Submit answers to `/supervisor` for feedback
4. Optionally regenerate quiz with supervisor feedback as focus comment
