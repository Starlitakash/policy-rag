"""
rag package
-----------
Exposes the core building blocks of the AI-Based Policy QA RAG system using Qdrant and Groq.
"""

from rag.config import (
    DOCUMENTS_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_STORAGE_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    GROQ_MODEL,
    GROQ_API_KEY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION
)
from rag.document_loader import load_pdf_file, load_pdf_directory, clean_text
from rag.text_splitter import split_documents, get_text_splitter
from rag.embeddings import get_embedding_model
from rag.vector_store import VectorStoreManager
from rag.retriever import PolicyRetriever
from rag.qa_chain import PolicyQAEngine, get_groq_llm

__all__ = [
    "load_pdf_file",
    "load_pdf_directory",
    "clean_text",
    "split_documents",
    "get_text_splitter",
    "get_embedding_model",
    "VectorStoreManager",
    "PolicyRetriever",
    "PolicyQAEngine",
    "get_groq_llm",
    "DOCUMENTS_DIR",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_STORAGE_PATH",
    "COLLECTION_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "GROQ_MODEL",
    "GROQ_API_KEY",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSION"
]
