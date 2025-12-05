import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from agents.sessions import session_manager
from agents.summarizer import summarize_pdf
from agents.examiner import generate_quiz
from agents.supervisor import provide_feedback, chat_with_supervisor

app = FastAPI(
    title="AI Tutor API",
    description="Multi-agent tutoring system with summarization, quiz generation, and Socratic feedback"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ExaminerRequest(BaseModel):
    session_id: str
    num_questions: int = 5


class SupervisorRequest(BaseModel):
    session_id: str
    message: str
    user_answers: Optional[List[str]] = None


class SummaryResponse(BaseModel):
    session_id: str
    summary: str


class QuizResponse(BaseModel):
    quiz: List[dict]


class SupervisorResponse(BaseModel):
    response: str
    messages: List[dict]


class SessionResponse(BaseModel):
    session_id: str
    has_summary: bool
    has_quiz: bool
    message_count: int


@app.post("/summarizer", response_model=SummaryResponse)
async def summarize_document(file: UploadFile = File(...)):
    """
    Upload a PDF and get a summary.
    Creates a new session and returns session_id with the summary.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    safe_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        session = session_manager.create_session(file_path)
        summary = summarize_pdf(file_path)
        session_manager.update_summary(session.session_id, summary)
        
        return SummaryResponse(
            session_id=session.session_id,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/examiner", response_model=QuizResponse)
async def generate_quiz_endpoint(request: ExaminerRequest):
    """
    Generate a quiz based on a previously summarized document.
    Requires a valid session_id from /summarizer.
    """
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.summary:
        raise HTTPException(status_code=400, detail="No summary found. Call /summarizer first.")
    
    if not session.docs:
        raise HTTPException(status_code=400, detail="Document not loaded")
    
    try:
        quiz = generate_quiz(
            docs=session.docs,
            summary=session.summary,
            num_questions=request.num_questions
        )
        session_manager.update_quiz(request.session_id, quiz)
        
        quiz_data = [task.model_dump() for task in quiz.tasks]
        return QuizResponse(quiz=quiz_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/supervisor", response_model=SupervisorResponse)
async def supervisor_chat(request: SupervisorRequest):
    """
    Chat with the Socratic tutor supervisor.
    
    First call should include user_answers to get initial feedback.
    Subsequent calls can just include message for follow-up questions.
    """
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.docs:
        raise HTTPException(status_code=400, detail="Document not loaded")
    
    if not session.summary:
        raise HTTPException(status_code=400, detail="No summary found")
    
    try:
        if request.user_answers:
            if not session.quiz:
                raise HTTPException(status_code=400, detail="No quiz found. Call /examiner first.")
            
            session_manager.update_user_answers(request.session_id, request.user_answers)
            
            response = provide_feedback(
                docs=session.docs,
                summary=session.summary,
                quiz=session.quiz,
                user_answers=request.user_answers
            )
        else:
            response = chat_with_supervisor(
                docs=session.docs,
                summary=session.summary,
                user_message=request.message,
                conversation_history=session.messages
            )
        
        session_manager.add_message(request.session_id, "user", request.message)
        session_manager.add_message(request.session_id, "assistant", response)
        
        return SupervisorResponse(
            response=response,
            messages=session_manager.get_messages(request.session_id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str):
    """Get information about a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session.session_id,
        has_summary=bool(session.summary),
        has_quiz=session.quiz is not None,
        message_count=len(session.messages)
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up resources."""
    if session_manager.delete_session(session_id):
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
