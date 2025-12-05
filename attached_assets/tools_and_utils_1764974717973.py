from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from agents.model import llm
from dotenv import load_dotenv

load_dotenv()

class Docs:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vector_store = self.upload_file(file_path)
        map_prompt = ChatPromptTemplate.from_template(
            "Summarize the following text in 3–5 short bullet points.\n\n{chunk}"
        )
        self.map_chain = map_prompt | llm


    def upload_file(self, file_path: str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)

        return vector_store

    def as_search_tool(self):
        @tool(name="search_in_docs", response_format="content_and_artifact")
        def _search_in_docs(query: str):
            """Retrieve information from the uploaded document to answer a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return _search_in_docs

    def get_diverse_chunks_mmr(self, query: str, k: int = 30):
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                # how much to prefer diversity vs similarity (0–1)
                "lambda_mult": 0.5,
                # optional: fetch more before pruning for diversity
                "fetch_k": max(k * 3, 50),
            },
        )
        return retriever.invoke(query)

    def summarize_chunks(self, k: int = 30):
        chunks = self.get_diverse_chunks_mmr("Give an overview of the main ideas in this document", k=k)

        for chunk in chunks:
            self.map_chain.