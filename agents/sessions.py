import uuid
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from agents.tools import Docs
from agents.states import Quiz


@dataclass
class Session:
    """Represents a single user session with all associated data."""
    session_id: str
    file_path: str
    docs: Optional[Docs] = None
    summary: str = ""
    quiz: Optional[Quiz] = None
    user_answers: List[str] = field(default_factory=list)
    messages: List[dict] = field(default_factory=list)


class SessionManager:
    """Manages user sessions for the AI Tutor API."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
    
    def create_session(self, file_path: str) -> Session:
        """
        Create a new session with the given PDF file.
        
        Args:
            file_path: Path to the saved PDF file
            
        Returns:
            New Session object
        """
        session_id = str(uuid.uuid4())
        docs = Docs(file_path)
        session = Session(
            session_id=session_id,
            file_path=file_path,
            docs=docs
        )
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up resources.
        
        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def update_summary(self, session_id: str, summary: str) -> bool:
        """Update the summary for a session."""
        session = self.get_session(session_id)
        if session:
            session.summary = summary
            return True
        return False
    
    def update_quiz(self, session_id: str, quiz: Quiz) -> bool:
        """Update the quiz for a session."""
        session = self.get_session(session_id)
        if session:
            session.quiz = quiz
            return True
        return False
    
    def update_user_answers(self, session_id: str, answers: List[str]) -> bool:
        """Update user answers for a session."""
        session = self.get_session(session_id)
        if session:
            session.user_answers = answers
            return True
        return False
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to the conversation history."""
        session = self.get_session(session_id)
        if session:
            session.messages.append({"role": role, "content": content})
            return True
        return False
    
    def get_messages(self, session_id: str) -> List[dict]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        if session:
            return session.messages
        return []


session_manager = SessionManager()
