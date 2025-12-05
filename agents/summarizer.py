from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List
from math import ceil
import time

load_dotenv()


def summarize_pdf(
    pdf_path: str,
    chunk_size: int = 6000,
    chunk_overlap: int = 100,
    batch_size: int = 4,
    delay_between_batches: float = 0.4,
) -> str:
    """
    Fast long-PDF summarizer using map-reduce approach.

    - Loads and chunks PDF
    - Summarizes chunks in parallel batches (map step)
    - Combines all partial summaries into one final summary (reduce step)
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        batch_size: Number of chunks to process in parallel
        delay_between_batches: Delay between API calls to avoid rate limits
    
    Returns:
        Final summary string
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs: List[Document] = splitter.split_documents(docs)

    llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")

    map_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 3-5 short bullet points.\n\n{chunk}"
    )
    map_chain = map_prompt | llm

    inputs = [{"chunk": d.page_content} for d in split_docs]
    partial_summaries = []
    num_batches = ceil(len(inputs) / batch_size)

    for b in range(num_batches):
        batch = inputs[b * batch_size: (b + 1) * batch_size]
        responses = map_chain.batch(batch)
        partial_summaries.extend(r.content for r in responses)
        if b < num_batches - 1 and delay_between_batches > 0:
            time.sleep(delay_between_batches)

    reduce_prompt = ChatPromptTemplate.from_template(
        "You are combining partial summaries of a long document.\n"
        "Write a concise final summary (max ~300 words) with clear sections if useful.\n\n"
        "Partial summaries:\n{partials}"
    )
    reduce_chain = reduce_prompt | llm

    final = reduce_chain.invoke(
        {"partials": "\n\n".join(partial_summaries)}
    )

    return final.content
