"""
rag/vector_store.py
-------------------
Manages Qdrant vector storage, collection indexing, similarity searches,
and metadata tracking for policy documents.

Supports both:
  1. Qdrant Cloud / Remote Server: via `url` and `api_key`
  2. Local File Storage: via `storage_path` (e.g. `./qdrant_db`)

Concept:
  Qdrant is a high-performance vector similarity search engine and database.
  It stores points consisting of:
  - Vector: 384-dimensional dense float vector
  - Payload: Document text + structured metadata (source filename, page, chunk_id)
  - ID: Unique identifier for each point
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_STORAGE_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION
)
from rag.embeddings import get_embedding_model


class VectorStoreManager:
    """
    Manages Qdrant vector storage (Cloud or Local), collection lifecycle,
    and similarity search operations.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: Optional[Embeddings] = None
    ):
        self.url = (url if url is not None else QDRANT_URL).strip()
        self.api_key = (api_key if api_key is not None else QDRANT_API_KEY).strip()
        self.storage_path = storage_path or QDRANT_STORAGE_PATH
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client (Cloud or Local)."""
        if self._client is None:
            if self.url:
                # Connect to Qdrant Cloud or remote cluster
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key if self.api_key else None
                )
            else:
                # Connect to local file-based persistent storage
                Path(self.storage_path).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=self.storage_path)

            self._ensure_collection_exists()
        return self._client

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection with Cosine distance if it doesn't already exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            if "Forbidden" in str(e) or "Unauthorized" in str(e) or "401" in str(e) or "403" in str(e):
                raise PermissionError(
                    f"Authentication error connecting to Qdrant Cloud ({self.url}). "
                    "Please provide a valid QDRANT_API_KEY in your .env file."
                ) from e
            raise e

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Lazy initialization of the LangChain QdrantVectorStore wrapper."""
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_model
            )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> int:
        """
        Adds or updates document chunks in the Qdrant vector database.
        Returns the number of documents added.
        """
        if not documents:
            return 0

        self.vector_store.add_documents(documents=documents)
        return len(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Executes a semantic similarity search in Qdrant for a given query string.
        Returns the top-k most relevant Document objects.
        """
        return self.vector_store.similarity_search(query=query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Executes similarity search and returns both the Document and its Cosine similarity score.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=k)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns statistics and metadata about the indexed policy documents in Qdrant.
        """
        mode = "cloud" if self.url else "local"
        location = self.url if self.url else self.storage_path

        try:
            if not self.client.collection_exists(self.collection_name):
                return {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "documents": [],
                    "collection_name": self.collection_name,
                    "mode": mode,
                    "location": location
                }

            collection_info = self.client.get_collection(self.collection_name)
            total_chunks = collection_info.points_count or 0

            # Scroll through points to summarize distinct files and indexed pages
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            distinct_sources: Dict[str, set] = {}
            for point in points:
                payload = point.payload or {}
                meta = payload.get("metadata", payload)
                source = meta.get("source")
                page = meta.get("page", 1)

                if source:
                    if source not in distinct_sources:
                        distinct_sources[source] = set()
                    distinct_sources[source].add(page)

            files_summary = [
                {
                    "source": src,
                    "indexed_pages": len(pages),
                    "pages_list": sorted(list(pages))
                }
                for src, pages in distinct_sources.items()
            ]

            return {
                "total_chunks": total_chunks,
                "total_documents": len(distinct_sources),
                "documents": files_summary,
                "collection_name": self.collection_name,
                "mode": mode,
                "location": location
            }

        except Exception as e:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "documents": [],
                "mode": mode,
                "location": location,
                "error": str(e)
            }

    def reset_collection(self) -> bool:
        """
        Deletes and re-creates the collection to clear all indexed documents.
        """
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            self._vector_store = None
            return True
        except Exception as e:
            print(f"Error resetting Qdrant collection: {e}")
            return False


if __name__ == "__main__":
    from rag.document_loader import load_pdf_directory
    from rag.text_splitter import split_documents
    from rag.config import DOCUMENTS_DIR

    print("Initializing Qdrant VectorStoreManager...")
    manager = VectorStoreManager()

    stats = manager.get_collection_stats()
    print(f"Connected to Qdrant Mode: {stats['mode']} ({stats['location']})")
    print(f"Total Points in Collection '{manager.collection_name}': {stats['total_chunks']}")
