"""
rag/embeddings.py
-----------------
Embeddings generator using FastEmbed for the Qdrant vector database.

Concept:
  An embedding model maps text passages and search queries into dense numerical
  vectors (e.g., 384 floating-point numbers) in a continuous semantic vector space.

Why FastEmbed + Qdrant:
  FastEmbed is built and maintained by Qdrant specifically for embedding generation.
  It runs locally using ONNX Runtime:
  - Zero API key requirements
  - Extremely fast CPU execution
  - Generates 384-dimensional dense vectors (BAAI/bge-small-en-v1.5)
  - Ideal for Qdrant's Cosine distance indexing
"""

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from rag.config import EMBEDDING_MODEL_NAME


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> Embeddings:
    """
    Returns an instance of FastEmbedEmbeddings using the specified model name.
    Default: 'BAAI/bge-small-en-v1.5' (384 dimensions).
    """
    try:
        return FastEmbedEmbeddings(model_name=model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize FastEmbed embeddings. Error: {e}"
        )


if __name__ == "__main__":
    print("Testing FastEmbed Embeddings model...")
    embedder = get_embedding_model()
    sample_text = "What is the university attendance requirement?"
    vec = embedder.embed_query(sample_text)
    print(f"[OK] Generated embedding vector!")
    print(f"Embedding Vector Dimensions: {len(vec)}")
    print(f"First 5 vector values: {vec[:5]}")
