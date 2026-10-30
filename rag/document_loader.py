"""
rag/document_loader.py
----------------------
Handles loading PDF documents and extracting raw text page-by-page.

Concept:
  A Document Loader reads raw files (e.g. PDF, DOCX, TXT) from disk and converts
  them into standardized LangChain `Document` objects containing `page_content` (the text)
  and `metadata` (source filename, page number, file size, etc.).

Why We Need It:
  LLMs cannot read binary PDF files directly. We must extract the plain text while
  preserving the page numbers and filenames so we can provide accurate citations later.
"""

import os
import re
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text:
    - Normalizes unicode characters and line breaks
    - Removes excessive repeated whitespace
    - Preserves meaningful paragraphs
    """
    if not text:
        return ""
    # Replace carriage returns and excessive whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace 3 or more newlines with 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def load_pdf_file(file_path: str) -> List[Document]:
    """
    Loads a single PDF file and returns a list of LangChain Document objects,
    one Document per page, with clean metadata.
    
    Metadata attached:
      - source: base filename (e.g., "SRITR_Academic_Attendance_Policy.pdf")
      - file_path: absolute path to file
      - page: page number (1-indexed for human-friendly citations)
      - total_pages: total number of pages in the PDF
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File {file_path} is not a PDF file.")

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    documents: List[Document] = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        # Only add documents that actually have content
        if cleaned:
            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": path.name,
                    "file_path": str(path.resolve()),
                    "page": page_idx + 1,  # 1-indexed
                    "total_pages": total_pages,
                }
            )
            documents.append(doc)

    return documents


def load_pdf_directory(directory_path: str) -> List[Document]:
    """
    Loads all PDF documents located in a given directory.
    Returns a combined flat list of LangChain Document objects across all PDFs.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    all_documents: List[Document] = []
    pdf_files = sorted(list(dir_path.glob("*.pdf")))

    for pdf_file in pdf_files:
        docs = load_pdf_file(str(pdf_file))
        all_documents.extend(docs)

    return all_documents


if __name__ == "__main__":
    # Quick self-test demonstration
    from rag.config import DOCUMENTS_DIR
    print(f"Loading PDFs from: {DOCUMENTS_DIR}")
    loaded_docs = load_pdf_directory(DOCUMENTS_DIR)
    print(f"Successfully loaded {len(loaded_docs)} pages across all PDF documents.")
    if loaded_docs:
        print("\n--- Sample Loaded Page ---")
        print(f"Source: {loaded_docs[0].metadata['source']} (Page {loaded_docs[0].metadata['page']})")
        print(f"Content preview:\n{loaded_docs[0].page_content[:200]}...")
