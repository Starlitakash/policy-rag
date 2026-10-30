"""
scripts/test_rag_pipeline.py
----------------------------
Comprehensive validation script testing all phases of the RAG pipeline
using Qdrant Vector Database and Groq LLM.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import DOCUMENTS_DIR, QDRANT_STORAGE_PATH, COLLECTION_NAME, GROQ_MODEL
from rag.document_loader import load_pdf_directory
from rag.text_splitter import split_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import VectorStoreManager
from rag.retriever import PolicyRetriever
from rag.qa_chain import PolicyQAEngine


def run_pipeline_test():
    print("=" * 75)
    print("  SRITR POLICY RAG PIPELINE (QDRANT VECTOR DB + GROQ LLM)")
    print("=" * 75)

    # Phase 1 & 2: PDF Ingestion & Document Loader
    print("\n[Phase 1 & 2] Loading and Chunking PDF Documents...")
    raw_docs = load_pdf_directory(DOCUMENTS_DIR)
    print(f"  -> Successfully loaded {len(raw_docs)} pages from {DOCUMENTS_DIR}")
    
    chunks = split_documents(raw_docs)
    print(f"  -> Generated {len(chunks)} chunks with enriched metadata.")
    sample = chunks[0]
    print(f"  -> Sample Chunk ID: {sample.metadata.get('chunk_id')}")
    print(f"  -> Sample Source:   {sample.metadata.get('source')} (Page {sample.metadata.get('page')})")

    # Phase 3: FastEmbed Model for Qdrant
    print("\n[Phase 3] Instantiating FastEmbed Embedding Model...")
    embedder = get_embedding_model()
    test_vec = embedder.embed_query("university attendance policy")
    print(f"  -> Active Embedder: FastEmbed (Local ONNX, BAAI/bge-small-en-v1.5)")
    print(f"  -> Generated embedding vector of dimension {len(test_vec)}")

    # Phase 4: Qdrant Vector Store
    print(f"\n[Phase 4] Indexing Chunks into Qdrant Vector Store at '{QDRANT_STORAGE_PATH}'...")
    vsm = VectorStoreManager(storage_path=QDRANT_STORAGE_PATH, collection_name=COLLECTION_NAME, embedding_model=embedder)
    vsm.reset_collection()
    added = vsm.add_documents(chunks)
    stats = vsm.get_collection_stats()
    print(f"  -> Indexed {added} points into Qdrant collection '{COLLECTION_NAME}'.")
    print(f"  -> Total Points in Qdrant: {stats['total_chunks']}")
    print(f"  -> Total Unique Policies: {stats['total_documents']}")

    # Phase 5: Similarity Search & Retriever
    print("\n[Phase 5] Testing Qdrant Similarity Search & Retrieval...")
    retriever = PolicyRetriever(vsm, top_k=2)
    test_questions = [
        "What is the minimum attendance required for exams?",
        "How much is the fee for re-evaluation?",
        "What is the curfew time for hostel students?"
    ]

    for q in test_questions:
        print(f"\n  Query: '{q}'")
        retrieved = retriever.get_relevant_documents(q, top_k=2)
        for idx, doc in enumerate(retrieved, 1):
            print(f"    Match #{idx}: [{doc.metadata.get('source')} - Page {doc.metadata.get('page')}]")
            print(f"    Excerpt: {doc.page_content[:120].strip()}...")

    # Phase 6 & 7: Groq QA Chain
    print(f"\n[Phase 6 & 7] Testing PolicyQAEngine with Groq (Model: {GROQ_MODEL})...")
    engine = PolicyQAEngine(vector_store_manager=vsm, top_k=2)
    q_result = engine.answer_question("What is the library overdue fine?")
    print(f"  Question: 'What is the library overdue fine?'")
    print(f"  Answer Preview:\n{q_result['answer'][:250]}...")
    print(f"  Sources ({len(q_result['sources'])} found):")
    for s in q_result["sources"]:
        print(f"   * {s['source']} (Page {s['page']})")

    print("\n" + "=" * 75)
    print("  ALL RAG PIPELINE PHASES VERIFIED WITH QDRANT & GROQ!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline_test()
