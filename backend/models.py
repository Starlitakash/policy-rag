"""
backend/models.py
-----------------
Pydantic schemas defining request and response models for FastAPI endpoints
using Qdrant and Groq.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request schema for asking a question about policy documents."""
    question: str = Field(
        ...,
        description="The question to ask about university policies.",
        example="What is the minimum attendance required to appear in examinations?"
    )
    top_k: Optional[int] = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of relevant document chunks to retrieve from Qdrant."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional Groq API key override."
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional Groq model name (default: llama-3.1-8b-instant)."
    )


class SourceItem(BaseModel):
    """Citation schema for retrieved document source from Qdrant."""
    source: str = Field(..., description="Filename of the policy document (e.g. 'Academic_Attendance_Policy.pdf')")
    page: int = Field(..., description="Page number where the excerpt appears")
    chunk_id: str = Field(..., description="Unique ID for the chunk")
    excerpt: str = Field(..., description="Text snippet from the document chunk")


class AskResponse(BaseModel):
    """Response schema returned by the /ask endpoint."""
    question: str
    answer: str
    sources: List[SourceItem]
    context_used: Optional[str] = None


class DocumentInfo(BaseModel):
    """Summary information for an indexed policy document in Qdrant."""
    source: str
    indexed_pages: int
    pages_list: List[int]


class DocumentsResponse(BaseModel):
    """Response schema returned by the /documents endpoint."""
    total_chunks: int
    total_documents: int
    documents: List[DocumentInfo]
    collection_name: str
    storage_path: str


class UploadResponse(BaseModel):
    """Response schema returned by the /upload endpoint."""
    message: str
    uploaded_files: List[str]
    pages_extracted: int
    chunks_created: int
    total_collection_chunks: int


class HealthResponse(BaseModel):
    """Response schema returned by the /health endpoint."""
    status: str
    app_name: str
    version: str
    vector_database: str
    llm_provider: str
    default_model: str
    indexed_chunks: int
    indexed_documents: int
