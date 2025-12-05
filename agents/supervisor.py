from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from agents.states import Quiz
from agents.prompts import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT, SUPERVISOR_CHAT_PROMPT
from agents.tools import Docs
from typing import List, Optional, Any, Tuple, Union
from dotenv import load_dotenv

load_dotenv()


def create_supervisor_agent(docs: Docs):
    """
    Create a Supervisor agent for Socratic tutoring.
    
    Args:
        docs: Docs instance with loaded document
    
    Returns:
        A LangGraph ReAct agent configured for tutoring
    """
    llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")
    search_tool = docs.as_search_tool()
    
    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
    )
    
    return agent


def format_quiz_results(quiz: Quiz, user_answers: List[str]) -> str:
    """Format quiz results for the supervisor to review."""
    results = []
    correct_count = 0
    
    for i, (task, user_answer) in enumerate(zip(quiz.tasks, user_answers)):
        correct = task.correct_answer or ""
        is_correct = user_answer.strip().lower() == correct.strip().lower()
        if is_correct:
            correct_count += 1
        
        result = f"""
Question {task.task_id}: {task.task}
Type: {task.task_type}
"""
        if task.answer_options:
            result += f"Options: {', '.join(task.answer_options)}\n"
        result += f"""Student Answer: {user_answer}
Correct Answer: {task.correct_answer}
Result: {'CORRECT' if is_correct else 'INCORRECT'}
"""
        results.append(result)
    
    header = f"Score: {correct_count}/{len(quiz.tasks)} ({100*correct_count/len(quiz.tasks):.0f}%)\n"
    return header + "\n".join(results)


def provide_feedback(
    docs: Docs,
    summary: str,
    quiz: Quiz,
    user_answers: List[str]
) -> str:
    """
    Provide Socratic feedback on quiz performance.
    
    Args:
        docs: Docs instance with loaded document
        summary: Summary of the document
        quiz: The quiz that was taken
        user_answers: User's answers to the quiz
    
    Returns:
        Feedback string from the supervisor
    """
    llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")
    search_tool = docs.as_search_tool()
    
    quiz_results = format_quiz_results(quiz, user_answers)
    
    context_docs = docs.similarity_search("main concepts explanation", k=3)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        ("human", SUPERVISOR_USER_PROMPT + "\n\nRelevant Document Context:\n{context}")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "summary": summary,
        "quiz_results": quiz_results,
        "context": context
    })
    
    return response.content


def chat_with_supervisor(
    docs: Docs,
    summary: str,
    user_message: str,
    conversation_history: Optional[List[dict]] = None
) -> str:
    """
    Continue tutoring conversation with the supervisor.
    
    Args:
        docs: Docs instance with loaded document
        summary: Summary of the document
        user_message: User's message/question
        conversation_history: Previous conversation messages
    
    Returns:
        Supervisor's response
    """
    llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")
    
    context_docs = docs.similarity_search(user_message, k=3)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    
    messages: List[Tuple[str, str]] = [
        ("system", f"You are a Socratic tutor. Document Summary: {summary}\n\nRelevant Context:\n{context}")
    ]
    
    if conversation_history:
        for msg in conversation_history:
            messages.append((msg["role"], msg["content"]))
    
    messages.append(("human", user_message))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    
    response = chain.invoke({})
    
    return response.content
