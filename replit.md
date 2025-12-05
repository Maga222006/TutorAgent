# AI Tutor Multi-Agent System

## Overview
A LangChain-based multi-agent system for educational tutoring with three independent agents, exposed via a FastAPI REST API:
1. **Summarizer** - Summarizes large PDFs using token-efficient stuff/refine methods
2. **Examiner** - Creates formatted quiz questions with Pydantic schema validation
3. **Supervisor** - Provides Socratic feedback on quiz results with chat capabilities

All agents share access to a RAG search tool for document retrieval.

## Project Structure
```
├── agents/                  # Main agents package
│   ├── __init__.py         # Package exports
│   ├── model.py            # LLM configuration (Google Gemini)
│   ├── states.py           # Pydantic schemas (QuizTask, Quiz, WorkflowState)
│   ├── tools.py            # Docs class with RAG vector store
│   ├── prompts.py          # Prompt templates for all agents
│   ├── summarizer.py       # Token-efficient PDF summarization (stuff/refine)
│   ├── examiner.py         # Quiz generation with structured output
│   ├── supervisor.py       # Socratic tutoring feedback and chat
│   └── sessions.py         # Session management for API state
├── documents/              # PDF documents to process
│   └── Lecture3.pdf        # Sample NLP lecture on word embeddings
├── app.py                  # FastAPI REST API
├── main.py                 # Legacy terminal app (reference only)
└── pyproject.toml          # Python dependencies
```

## Dependencies
- langchain, langchain-community, langchain-core, langgraph
- langchain-google-genai (for Google Gemini LLM integration)
- langchain-huggingface (for sentence-transformers embeddings)
- fastapi, uvicorn, python-multipart (REST API)
- pypdf (PDF loading)
- pydantic (structured outputs)
- faiss-cpu, chromadb (vector stores)

## Environment Variables
- `GEMINI_API_KEY` - Required. Google Gemini API key for LLM access

## API Endpoints

### POST /summarizer
Upload a PDF and get a summary. Creates a new session.
- **Body**: Form data with `file` (PDF upload)
- **Returns**: `{session_id, summary}`

### POST /examiner
Generate a quiz based on a previously summarized document.
- **Body**: `{session_id, num_questions}`
- **Returns**: `{quiz: [...]}`

### POST /supervisor
Chat with the Socratic tutor. First call should include `user_answers` for initial feedback.
- **Body**: `{session_id, message, user_answers (optional)}`
- **Returns**: `{response, messages}`

### GET /session/{session_id}
Get session information.
- **Returns**: `{session_id, has_summary, has_quiz, message_count}`

### DELETE /session/{session_id}
Delete a session and clean up resources.

### GET /health
Health check endpoint.

## Usage
The server runs on port 5000. Start with:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

Example workflow:
1. Upload PDF to `/summarizer` → get session_id + summary
2. Generate quiz via `/examiner` with session_id
3. Submit answers to `/supervisor` for Socratic feedback
4. Continue chatting with `/supervisor` for further tutoring

## Architecture
- Uses Google Gemini with `gemini-2.0-flash-lite` model
- Vector store: InMemoryVectorStore with sentence-transformers embeddings
- Summarization: Token-efficient stuff method for small docs, refine for large docs
- Quiz generation: Structured output with Pydantic validation
- Session management: In-memory session storage with Docs, Quiz, and chat history

## Recent Changes
- 2025-12-05: Converted to FastAPI REST API with session management
- 2025-12-05: Added sessions.py for managing user sessions
- 2025-12-05: Created app.py with all API endpoints
- 2025-12-05: Changed model from Groq to Google Gemini
- 2025-12-05: Rewrote summarizer to be token-efficient
