# AI Tutor Multi-Agent System

## Overview
A LangChain-based multi-agent system for educational tutoring with three independent agents:
1. **Summarizer** - Summarizes large PDFs using map-reduce approach
2. **Examiner** - Creates formatted quiz questions with Pydantic schema validation
3. **Supervisor** - Provides Socratic feedback on quiz results

All agents share access to a RAG search tool for document retrieval.

## Project Structure
```
├── agents/                  # Main agents package
│   ├── __init__.py         # Package exports
│   ├── model.py            # LLM configuration (Groq + Llama)
│   ├── states.py           # Pydantic schemas (QuizTask, Quiz, WorkflowState)
│   ├── tools.py            # Docs class with RAG vector store
│   ├── prompts.py          # Prompt templates for all agents
│   ├── summarizer.py       # Map-reduce PDF summarization
│   ├── examiner.py         # Quiz generation with structured output
│   ├── supervisor.py       # Socratic tutoring feedback
│   └── workflow.py         # LangGraph workflow orchestration
├── documents/              # PDF documents to process
│   └── Lecture3.pdf        # Sample NLP lecture on word embeddings
├── main.py                 # Entry point with demo and interactive modes
└── pyproject.toml          # Python dependencies
```

## Dependencies
- langchain, langchain-community, langchain-core, langgraph
- langchain-google-genai (for Google Gemini LLM integration)
- langchain-huggingface (for sentence-transformers embeddings)
- pypdf (PDF loading)
- pydantic (structured outputs)
- faiss-cpu, chromadb (vector stores)

## Environment Variables
- `GEMINI_API_KEY` - Required. Google Gemini API key for LLM access

## Usage
Run `python main.py` to start. Choose:
1. Demo mode - Automated test with sample answers
2. Interactive mode - User provides PDF path and quiz answers

## Architecture
- Uses Google Gemini with `gemini-2.0-flash-lite` model
- Vector store: InMemoryVectorStore with sentence-transformers embeddings
- Summarization: Map-reduce pattern with batched parallel processing
- Quiz generation: Structured output with Pydantic validation
- Workflow: LangGraph StateGraph for agent orchestration

## Recent Changes
- 2025-12-05: Created complete multi-agent system with all components
