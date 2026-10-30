"""
backend/app.py
--------------
FastAPI backend providing REST API endpoints for document upload,
collection statistics, and RAG-based Question Answering using Qdrant and Groq.
"""

import os
import shutil
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag.config import (
    DOCUMENTS_DIR,
    QDRANT_STORAGE_PATH,
    COLLECTION_NAME,
    GROQ_MODEL
)
from rag.document_loader import load_pdf_file, load_pdf_directory
from rag.text_splitter import split_documents
from rag.vector_store import VectorStoreManager
from rag.qa_chain import PolicyQAEngine
from backend.models import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    UploadResponse,
    HealthResponse
)

# Initialize FastAPI application
app = FastAPI(
    title="AI-Based Policy QA API (Qdrant + Groq)",
    description=(
        "Production-grade RAG API for uploading policy documents "
        "and performing strictly grounded question answering using Qdrant vector database "
        "and Groq LLM with page-level citations."
    ),
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances of VectorStoreManager (Qdrant) and QAEngine (Groq)
vector_store_manager = VectorStoreManager()
qa_engine = PolicyQAEngine(vector_store_manager=vector_store_manager)


@app.get("/", summary="API Root & Overview")
async def root():
    """Returns welcoming overview and links to API documentation."""
    return {
        "project": "AI-Based QA for Policy Documents",
        "domain": "Enterprise & Organizational Policy Intelligence",
        "vector_database": "Qdrant",
        "llm_engine": "Groq",
        "default_model": GROQ_MODEL,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "POST /upload": "Upload and index new PDF policy documents into Qdrant",
            "POST /ask": "Ask questions answered strictly by uploaded documents using Groq",
            "GET /documents": "List all indexed policy documents and Qdrant stats",
            "POST /reindex": "Re-index all documents present in the documents folder",
            "POST /reset": "Reset and empty the Qdrant vector database collection",
            "GET /health": "System health and index status"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="System Health & Index Status")
async def health_check():
    """
    Returns system status, Qdrant vector database stats, and active Groq model.
    """
    stats = vector_store_manager.get_collection_stats()
    return HealthResponse(
        status="healthy",
        app_name="AI-Based QA for Policy Documents",
        version="2.0.0",
        vector_database="qdrant",
        llm_provider="groq",
        default_model=GROQ_MODEL,
        indexed_chunks=stats.get("total_chunks", 0),
        indexed_documents=stats.get("total_documents", 0)
    )


@app.get("/documents", response_model=DocumentsResponse, summary="List Indexed Documents")
async def list_documents():
    """
    Lists all university policy documents indexed in the Qdrant vector store,
    including total chunks, page counts, and storage location.
    """
    stats = vector_store_manager.get_collection_stats()
    return DocumentsResponse(
        total_chunks=stats.get("total_chunks", 0),
        total_documents=stats.get("total_documents", 0),
        documents=stats.get("documents", []),
        collection_name=stats.get("collection_name", COLLECTION_NAME),
        storage_path=stats.get("storage_path", QDRANT_STORAGE_PATH)
    )


@app.post("/upload", response_model=UploadResponse, summary="Upload & Index PDF Documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Uploads one or multiple PDF documents:
    1. Saves PDF files into the `documents/` directory
    2. Loads text page-by-page with page metadata
    3. Splits text into semantic chunks with unique chunk IDs
    4. Generates embeddings and stores points into Qdrant
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded."
        )

    saved_filenames = []
    all_extracted_docs = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are supported."
            )

        destination_path = Path(DOCUMENTS_DIR) / file.filename
        
        # Save file to disk
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_filenames.append(file.filename)

        # Load the newly saved PDF
        extracted_pages = load_pdf_file(str(destination_path))
        all_extracted_docs.extend(extracted_pages)

    # Chunk the extracted pages
    chunks = split_documents(all_extracted_docs)

    # Add chunks to Qdrant
    added_count = vector_store_manager.add_documents(chunks)
    total_stats = vector_store_manager.get_collection_stats()

    return UploadResponse(
        message=f"Successfully uploaded and indexed {len(saved_filenames)} document(s) in Qdrant.",
        uploaded_files=saved_filenames,
        pages_extracted=len(all_extracted_docs),
        chunks_created=len(chunks),
        total_collection_chunks=total_stats.get("total_chunks", added_count)
    )


@app.post("/ask", response_model=AskResponse, summary="Ask Question on Policy Documents")
async def ask_question(request: AskRequest):
    """
    Processes a user question through the RAG pipeline:
    1. Generates question embedding
    2. Retrieves top-K matching policy chunks from Qdrant
    3. Constructs strict grounding prompt
    4. Invokes Groq LLM (e.g. llama-3.1-8b-instant)
    5. Returns grounded answer and source citations
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    try:
        response = qa_engine.answer_question(
            question=request.question,
            top_k=request.top_k,
            api_key=request.api_key,
            model_name=request.model_name
        )

        return AskResponse(
            question=response["question"],
            answer=response["answer"],
            sources=response["sources"],
            context_used=response.get("context_used")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing QA chain: {str(e)}"
        )


@app.post("/reindex", summary="Re-index All Documents in Folder")
async def reindex_all():
    """
    Reads all PDF files present in the `documents/` folder, splits them into chunks,
    and updates the Qdrant vector database.
    """
    try:
        raw_docs = load_pdf_directory(DOCUMENTS_DIR)
        chunks = split_documents(raw_docs)
        vector_store_manager.add_documents(chunks)
        stats = vector_store_manager.get_collection_stats()
        return {
            "message": "Re-indexing complete in Qdrant.",
            "pages_indexed": len(raw_docs),
            "chunks_indexed": len(chunks),
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-index documents: {str(e)}"
        )


@app.post("/reset", summary="Reset Vector Database")
async def reset_database():
    """
    Clears all documents from the Qdrant collection.
    """
    success = vector_store_manager.reset_collection()
    return {
        "success": success,
        "message": "Qdrant collection has been reset and emptied."
    }


if __name__ == "__main__":
    import uvicorn
    from rag.config import BACKEND_HOST, BACKEND_PORT
    print(f"Starting Policy RAG FastAPI server with Qdrant + Groq on http://{BACKEND_HOST}:{BACKEND_PORT}...")
    uvicorn.run("backend.app:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
