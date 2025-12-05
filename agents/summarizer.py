from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from agents.model import llm
from typing import List


MAX_CONTEXT_CHARS = 100000


def summarize_pdf(pdf_path: str) -> str:
    """
    Token-efficient PDF summarizer.
    
    Strategy:
    1. If document is small enough, summarize in ONE call (stuff method)
    2. If larger, use iterative refinement with large chunks (fewer API calls)
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Final summary string
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    full_text = "\n\n".join(doc.page_content for doc in docs)
    
    if len(full_text) <= MAX_CONTEXT_CHARS:
        return _stuff_summarize(full_text)
    else:
        return _refine_summarize(full_text)


def _stuff_summarize(text: str) -> str:
    """Summarize entire document in one API call."""
    prompt = ChatPromptTemplate.from_template(
        "You are an expert summarizer. Read the following document and provide "
        "a comprehensive summary covering all key topics, concepts, and important details.\n\n"
        "Format your summary with:\n"
        "- A brief overview (2-3 sentences)\n"
        "- Main topics/sections with key points\n"
        "- Important definitions or concepts\n\n"
        "Document:\n{text}"
    )
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content


def _refine_summarize(text: str, chunk_size: int = 50000) -> str:
    """
    Iterative refinement for large documents.
    
    Uses fewer, larger chunks and refines the summary incrementally.
    This uses far fewer API calls than map-reduce.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=500,
    )
    chunks = splitter.split_text(text)
    
    first_prompt = ChatPromptTemplate.from_template(
        "You are an expert summarizer. Summarize the following content, "
        "capturing all key topics, concepts, and important details:\n\n{text}"
    )
    first_chain = first_prompt | llm
    summary = first_chain.invoke({"text": chunks[0]}).content
    
    if len(chunks) == 1:
        return summary
    
    refine_prompt = ChatPromptTemplate.from_template(
        "You have an existing summary of a document:\n\n"
        "EXISTING SUMMARY:\n{summary}\n\n"
        "Now incorporate the following additional content into the summary. "
        "Expand and refine the summary to include new information while keeping it coherent:\n\n"
        "NEW CONTENT:\n{new_content}\n\n"
        "Provide the updated comprehensive summary:"
    )
    refine_chain = refine_prompt | llm
    
    for chunk in chunks[1:]:
        response = refine_chain.invoke({
            "summary": summary,
            "new_content": chunk
        })
        summary = response.content
    
    return summary
