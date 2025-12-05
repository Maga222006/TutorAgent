from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


class Docs:
    """Document manager with vector store for RAG-based retrieval."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = self._upload_file(file_path)

    def _upload_file(self, file_path: str) -> InMemoryVectorStore:
        """Load PDF, chunk it, and create vector store."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        
        vector_store = InMemoryVectorStore(self.embeddings)
        vector_store.add_documents(documents=all_splits)
        
        return vector_store

    def as_search_tool(self):
        """Return a LangChain tool for searching the document."""
        vector_store = self.vector_store
        
        @tool
        def search_in_docs(query: str) -> str:
            """Retrieve information from the uploaded document to answer a query."""
            retrieved_docs = vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized
        
        return search_in_docs

    def get_diverse_chunks_mmr(self, query: str, k: int = 30):
        """Get diverse chunks using MMR (Maximal Marginal Relevance)."""
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "lambda_mult": 0.5,
                "fetch_k": max(k * 3, 50),
            },
        )
        return retriever.invoke(query)

    def similarity_search(self, query: str, k: int = 4):
        """Simple similarity search."""
        return self.vector_store.similarity_search(query, k=k)
