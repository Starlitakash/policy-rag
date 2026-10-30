"""
rag/retriever.py
----------------
Manages the retrieval interface for finding the most relevant policy passages for a user's question.

Concept:
  A Retriever is a LangChain abstraction that accepts an unstructured text query (the user's question),
  interacts with the Vector Store, and returns a list of matching `Document` objects.

How Similarity Search Works:
  1. Question -> Vector: The user's question is embedded into a high-dimensional vector.
  2. Math Comparison: The question vector is compared against all stored chunk vectors using
     Cosine Similarity:
       Similarity(A, B) = (A · B) / (||A|| * ||B||)
  3. Top-K Selection: The top `k` chunks with the highest similarity score are retrieved.

What the Retriever Returns:
  A list of `Document` objects, each containing:
  - `page_content`: The exact text of the relevant policy clause
  - `metadata`: `source` filename, `page` number, `chunk_id`
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from rag.config import TOP_K
from rag.vector_store import VectorStoreManager


class PolicyRetriever:
    """
    Retriever wrapper that provides high-level methods for retrieving context
    and formatting sources for the QA Chain.
    """

    def __init__(self, vector_store_manager: VectorStoreManager, top_k: int = TOP_K):
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k

    def get_relevant_documents(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieves the top-k most relevant policy document chunks for a question.
        """
        k = top_k or self.top_k
        return self.vector_store_manager.similarity_search(query=query, k=k)

    def get_relevant_documents_with_scores(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k relevant chunks along with distance scores.
        """
        k = top_k or self.top_k
        return self.vector_store_manager.similarity_search_with_score(query=query, k=k)

    def format_docs_for_prompt(self, docs: List[Document]) -> str:
        """
        Combines retrieved Document chunks into a clean, structured context string
        to be injected into the LLM prompt.
        
        Example format:
        [Document: SRITR_Academic_Attendance_Policy.pdf | Page: 1 | Chunk: ..._p1_c01]
        Content: ...
        """
        if not docs:
            return "No relevant policy documents found in the database."

        formatted_blocks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            
            block = (
                f"--- EXCERPT {i} [Source: {source} | Page: {page} | Chunk ID: {chunk_id}] ---\n"
                f"{doc.page_content.strip()}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def extract_sources_metadata(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Extracts clean, deduplicated citation metadata from retrieved documents.
        Used to render source cards in the UI and API response.
        """
        sources = []
        seen_chunk_ids = set()

        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if chunk_id:
                seen_chunk_ids.add(chunk_id)

            sources.append({
                "source": doc.metadata.get("source", "Unknown Document"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": chunk_id or "unknown",
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })

        return sources


if __name__ == "__main__":
    # Quick self-test demonstration
    manager = VectorStoreManager()
    retriever = PolicyRetriever(manager, top_k=2)

    query = "What is the hostel in-time and curfew policy?"
    print(f"Testing PolicyRetriever for query: '{query}'\n")

    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} documents.\n")

    context_str = retriever.format_docs_for_prompt(docs)
    print("--- Formatted Context for LLM ---")
    print(context_str[:400] + "...\n")

    sources = retriever.extract_sources_metadata(docs)
    print("--- Extracted Sources Metadata ---")
    print(sources)
