"""
rag/text_splitter.py
--------------------
Handles chunking documents into optimal, overlapping text passages for vector embedding.

Concept:
  Text splitting (chunking) breaks long documents or whole pages into smaller,
  semantically coherent passages (chunks) of a targeted character or token length.

Why We Need It:
  1. Embedding models have context length limits (e.g. 512 tokens).
  2. Granular chunks improve search relevance: retrieving a specific 5-sentence paragraph
     matching the query is much more precise than retrieving an entire 30-page document.
  3. Chunk Overlap ensures concepts that span the boundary between two chunks are
     not split and lost in the middle of a sentence.

Metadata Enrichment:
  Each chunk is tagged with:
  - `source`: document filename
  - `page`: page number
  - `chunk_id`: globally unique ID (e.g., "SRITR_Academic_Attendance_Policy.pdf_p1_c01")
  - `chunk_index`: sequential chunk counter
  - `char_count`: length of the chunk text
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """
    Instantiates a RecursiveCharacterTextSplitter.
    
    Recursive splitting tries splitting on paragraphs (`\\n\\n`), then sentences (`\\n`, `. `),
    and finally words (` `), keeping semantic units as intact as possible.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits a list of page-level Document objects into smaller chunks and
    enriches each chunk with precise, citation-friendly metadata.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    enriched_chunks: List[Document] = []
    
    # Track chunk index per (source, page) to generate readable chunk IDs
    page_chunk_counts = {}

    for idx, chunk in enumerate(raw_chunks):
        source = chunk.metadata.get("source", "unknown_doc")
        page = chunk.metadata.get("page", 1)
        
        page_key = f"{source}_p{page}"
        page_chunk_counts[page_key] = page_chunk_counts.get(page_key, 0) + 1
        chunk_num_on_page = page_chunk_counts[page_key]

        # Generate a clean, unique chunk ID
        # Example: "SRITR_Academic_Attendance_Policy.pdf_p1_c01"
        chunk_id = f"{source}_p{page}_c{chunk_num_on_page:02d}"

        # Merge new metadata with existing parent document metadata
        updated_metadata = dict(chunk.metadata)
        updated_metadata.update({
            "chunk_id": chunk_id,
            "chunk_index": idx + 1,
            "chunk_page_index": chunk_num_on_page,
            "char_count": len(chunk.page_content)
        })

        chunk.metadata = updated_metadata
        enriched_chunks.append(chunk)

    return enriched_chunks


if __name__ == "__main__":
    # Quick self-test demonstration
    from rag.document_loader import load_pdf_directory
    from rag.config import DOCUMENTS_DIR

    docs = load_pdf_directory(DOCUMENTS_DIR)
    chunks = split_documents(docs)

    print(f"Loaded {len(docs)} pages -> Generated {len(chunks)} text chunks.")
    if chunks:
        sample = chunks[0]
        print("\n--- Sample Enriched Chunk ---")
        print(f"Chunk ID: {sample.metadata['chunk_id']}")
        print(f"Source:   {sample.metadata['source']} | Page: {sample.metadata['page']}")
        print(f"Chars:    {sample.metadata['char_count']}")
        print(f"Content:\n{sample.page_content[:250]}...")
