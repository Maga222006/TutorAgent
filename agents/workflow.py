from typing import TypedDict, Annotated, Literal, Optional, List
from langgraph.graph import StateGraph, END
from agents.summarizer import summarize_pdf
from agents.examiner import generate_quiz
from agents.supervisor import provide_feedback, chat_with_supervisor
from agents.tools import Docs
from agents.states import Quiz


class TutorState(TypedDict):
    """State for the AI Tutor workflow."""
    pdf_path: str
    docs: Optional[Docs]
    summary: str
    quiz: Optional[Quiz]
    user_answers: Optional[List[str]]
    feedback: str
    current_stage: str
    messages: List[dict]


def ingest_document(state: TutorState) -> TutorState:
    """Load and index the PDF document."""
    pdf_path = state["pdf_path"]
    docs = Docs(pdf_path)
    return {
        **state,
        "docs": docs,
        "current_stage": "ingested"
    }


def summarize_document(state: TutorState) -> TutorState:
    """Summarize the document using map-reduce."""
    pdf_path = state["pdf_path"]
    summary = summarize_pdf(pdf_path)
    return {
        **state,
        "summary": summary,
        "current_stage": "summarized"
    }


def generate_quiz_node(state: TutorState) -> TutorState:
    """Generate quiz questions from the document."""
    docs = state["docs"]
    summary = state["summary"]
    quiz = generate_quiz(docs, summary, num_questions=5)
    return {
        **state,
        "quiz": quiz,
        "current_stage": "quiz_generated"
    }


def evaluate_answers(state: TutorState) -> TutorState:
    """Evaluate user answers and provide feedback."""
    docs = state["docs"]
    summary = state["summary"]
    quiz = state["quiz"]
    user_answers = state["user_answers"]
    
    feedback = provide_feedback(docs, summary, quiz, user_answers)
    return {
        **state,
        "feedback": feedback,
        "current_stage": "feedback_provided"
    }


def route_after_ingest(state: TutorState) -> str:
    """Route after document ingestion."""
    return "summarize"


def route_after_summarize(state: TutorState) -> str:
    """Route after summarization."""
    return "generate_quiz"


def route_after_quiz(state: TutorState) -> str:
    """Route after quiz generation - wait for user answers."""
    if state.get("user_answers"):
        return "evaluate"
    return END


def create_tutor_workflow():
    """Create the AI Tutor LangGraph workflow."""
    workflow = StateGraph(TutorState)
    
    workflow.add_node("ingest", ingest_document)
    workflow.add_node("summarize", summarize_document)
    workflow.add_node("generate_quiz", generate_quiz_node)
    workflow.add_node("evaluate", evaluate_answers)
    
    workflow.set_entry_point("ingest")
    
    workflow.add_edge("ingest", "summarize")
    workflow.add_edge("summarize", "generate_quiz")
    workflow.add_conditional_edges(
        "generate_quiz",
        route_after_quiz,
        {
            "evaluate": "evaluate",
            END: END
        }
    )
    workflow.add_edge("evaluate", END)
    
    return workflow.compile()


class AITutor:
    """High-level interface for the AI Tutor system."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.docs = None
        self.summary = None
        self.quiz = None
        self.feedback = None
        self.conversation_history = []
    
    def ingest_and_summarize(self) -> str:
        """Load document and generate summary."""
        print("Loading document...")
        self.docs = Docs(self.pdf_path)
        print("Generating summary...")
        self.summary = summarize_pdf(self.pdf_path)
        return self.summary
    
    def generate_quiz(self, num_questions: int = 5) -> Quiz:
        """Generate quiz questions."""
        if not self.docs or not self.summary:
            raise ValueError("Must call ingest_and_summarize() first")
        
        print(f"Generating {num_questions} quiz questions...")
        self.quiz = generate_quiz(self.docs, self.summary, num_questions)
        return self.quiz
    
    def submit_answers(self, answers: List[str]) -> str:
        """Submit quiz answers and get feedback."""
        if not self.quiz:
            raise ValueError("Must call generate_quiz() first")
        
        print("Evaluating answers...")
        self.feedback = provide_feedback(
            self.docs, self.summary, self.quiz, answers
        )
        return self.feedback
    
    def chat(self, message: str) -> str:
        """Continue tutoring conversation."""
        if not self.docs or not self.summary:
            raise ValueError("Must call ingest_and_summarize() first")
        
        response = chat_with_supervisor(
            self.docs, self.summary, message, self.conversation_history
        )
        
        self.conversation_history.append({"role": "human", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
