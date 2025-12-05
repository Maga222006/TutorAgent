from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from agents.states import Quiz
from agents.prompts import EXAMINER_SYSTEM_PROMPT, EXAMINER_USER_PROMPT
from agents.tools import Docs
from agents.model import llm


def create_examiner_agent(docs: Docs):
    """
    Create an Examiner agent that generates quiz questions.
    
    Args:
        docs: Docs instance with loaded document
    
    Returns:
        A LangGraph ReAct agent configured for quiz generation
    """
    search_tool = docs.as_search_tool()
    
    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
    )
    
    return agent


def generate_quiz(docs: Docs, summary: str, num_questions: int = 5) -> Quiz:
    """
    Generate a quiz based on the document and summary.
    
    Args:
        docs: Docs instance with loaded document
        summary: Summary of the document
        num_questions: Number of questions to generate
    
    Returns:
        Quiz object with generated questions
    """
    llm_with_structure = llm.with_structured_output(Quiz)
    
    search_tool = docs.as_search_tool()
    
    context_docs = docs.similarity_search("main concepts and key topics", k=5)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXAMINER_SYSTEM_PROMPT),
        ("human", EXAMINER_USER_PROMPT + "\n\nAdditional Context from Document:\n{context}")
    ])
    
    chain = prompt | llm_with_structure
    
    quiz = chain.invoke({
        "summary": summary,
        "num_questions": num_questions,
        "context": context
    })
    
    return quiz
