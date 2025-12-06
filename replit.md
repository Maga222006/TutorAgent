# AI Tutor API

## Overview

This is a FastAPI backend application that provides an AI-powered tutoring system. It uses LangChain, LangGraph, and Google Gemini AI to create a multi-agent system for educational purposes.

## Features

- **PDF Summarization**: Upload PDF documents and get AI-generated summaries
- **Quiz Generation**: Automatically generate quizzes based on document content
- **Socratic Tutoring**: Interactive chat with an AI tutor that guides learning through questions
- **Content Safety**: Built-in guardrails to ensure age-appropriate content for all learners

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/summarizer` | POST | Upload a PDF and receive a summary |
| `/examiner` | POST | Generate a quiz from a summarized document |
| `/supervisor` | POST | Chat with the Socratic tutor for feedback |
| `/session/{session_id}` | GET | Retrieve session information |
| `/session/{session_id}` | DELETE | Delete a session |
| `/health` | GET | Health check endpoint |

## Project Structure

```
/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── agents/               # Multi-agent system modules
│   ├── prompts.py        # System prompts with content safety guardrails
│   ├── supervisor.py     # Socratic tutor agent
│   ├── examiner.py       # Quiz generation agent
│   ├── summarizer.py     # PDF summarization agent
│   ├── model.py          # LLM initialization (Google Gemini)
│   ├── sessions.py       # Session management
│   ├── states.py         # Pydantic models
│   └── tools.py          # LangChain tools
└── documents/            # Upload directory for PDFs
```

## Environment Variables

- `GOOGLE_API_KEY` (required): Google Gemini API key for AI functionality

## Recent Changes

- Added content safety guardrails to all prompts to protect young learners
- Fixed curly brace escaping bug that caused errors with mathematical notation in PDFs
- Configured for deployment on Replit

## Running the Application

The application runs on port 8000 with:
```
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Development Notes

- This is a backend-only API; no frontend is included
- Uses LangGraph for multi-agent orchestration
- Embeddings use HuggingFace sentence-transformers for document similarity search
