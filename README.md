# Policy RAG

A retrieval-augmented question-answering app for policy PDFs. It extracts PDF text page by page, splits it into overlapping passages, embeds those passages locally with FastEmbed, and stores them in Qdrant. Questions retrieve matching passages and, when a Groq API key is configured, send them to a Groq chat model for a context-grounded answer with source citations.

The project includes a Streamlit user interface and a FastAPI REST API. Sample policy PDFs are provided in `documents/`.

## Features

- Upload PDF policy documents and index them in Qdrant.
- Search indexed content with semantic similarity retrieval.
- Return answers with document, page, chunk, and excerpt citations.
- Use local persistent Qdrant storage or a configured Qdrant server.
- Configure the Groq model, embedding model, chunk size, overlap, and retrieval count with environment variables.
- Ask questions through the Streamlit interface or REST API.

The answer prompt instructs the model to use only retrieved policy context and to say when that context does not contain an answer. As with any generative model, verify important decisions against the cited source documents.

## How it works

1. `rag/document_loader.py` extracts text and page metadata from PDFs.
2. `rag/text_splitter.py` creates overlapping chunks and citation-friendly IDs.
3. `rag/embeddings.py` creates 384-dimensional embeddings with FastEmbed (`BAAI/bge-small-en-v1.5` by default).
4. `rag/vector_store.py` stores and searches chunks in Qdrant.
5. `rag/retriever.py` selects the top matching passages and formats citations.
6. `rag/qa_chain.py` sends the retrieved context and question to Groq.
7. `backend/app.py` exposes the pipeline as an API; `frontend/app.py` provides the Streamlit UI.

## Requirements

- Python with support for the dependencies in `requirements.txt`
- A Groq API key for generated answers
- Internet access on first use to download the local embedding model
- Optional: Qdrant Cloud or another Qdrant server. Without `QDRANT_URL`, the app uses local file storage.

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local environment file from the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your Groq API key:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

Do not commit `.env` or put API keys in source control. The app defaults to local Qdrant storage at `./qdrant_db`. To use a remote Qdrant instance, set `QDRANT_URL` and, if required, `QDRANT_API_KEY` in `.env`.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | unset | Groq key for generated answers |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq chat model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model used for document and query embeddings |
| `QDRANT_URL` | unset | Remote Qdrant URL; if empty, local storage is used |
| `QDRANT_API_KEY` | unset | Optional Qdrant server key |
| `QDRANT_STORAGE_PATH` | `./qdrant_db` | Local Qdrant data directory |
| `COLLECTION_NAME` | `sritr_university_policies` | Qdrant collection |
| `CHUNK_SIZE` | `800` | Chunk length in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between text chunks |
| `TOP_K` | `4` | Number of chunks retrieved for a question |
| `BACKEND_HOST` | `127.0.0.1` | FastAPI bind address |
| `BACKEND_PORT` | `8000` | FastAPI port |

If `GROQ_API_KEY` is not set, retrieval can still run, but generated answers are unavailable; the QA engine returns a notice and a preview of retrieved context instead.

## Run the application

### 1. Start the API

In an activated virtual environment:

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 2. Start the Streamlit UI

Open a second terminal in the project directory, activate the same virtual environment, and run:

```powershell
streamlit run frontend/app.py
```

Open the URL printed by Streamlit (typically `http://localhost:8501`). By default, the UI connects to the API at `http://127.0.0.1:8000`. Set `BACKEND_URL` if the API runs at a different address. The UI can also use the RAG pipeline directly if the backend is unavailable.

### 3. Index policy PDFs

Upload PDFs from the **Upload Policies** tab, or place PDFs in `documents/` and use the UI's **Re-Index Knowledge Base** action. The REST API also provides `POST /reindex` to index the PDFs already in that directory.

The included Qdrant directory, if present, contains local index state. To rebuild the index from the PDFs in `documents/`, use the re-index action or endpoint.

## REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | API overview |
| `GET` | `/health` | Service and index status |
| `GET` | `/documents` | List indexed documents and collection statistics |
| `POST` | `/upload` | Upload one or more PDF files as multipart form data under `files` |
| `POST` | `/ask` | Ask a question about indexed policies |
| `POST` | `/reindex` | Index PDFs currently in the `documents/` directory |
| `POST` | `/reset` | Clear the Qdrant collection |

Example question request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{"question":"What is the minimum attendance requirement?","top_k":4}'
```

The `/ask` response contains the question, answer, source citations, and retrieved context. `top_k` is optional and must be between 1 and 10. The optional `api_key` and `model_name` request fields override the configured Groq settings for that request.

## Project layout

```text
backend/       FastAPI app and request/response models
documents/     Sample and user-provided policy PDFs
frontend/      Streamlit app and styles
rag/           PDF loading, chunking, embeddings, retrieval, Qdrant, and QA chain
scripts/       PDF generation utilities and pipeline test script
qdrant_db/     Local persistent Qdrant data (when local storage is enabled)
```

## Run the pipeline script

The repository includes an end-to-end pipeline script:

```powershell
python scripts/test_rag_pipeline.py
```

This script resets and repopulates the configured Qdrant collection before running retrieval and QA checks. **It deletes the existing contents of that collection**; use a test collection or back up important data before running it against a collection you need to keep.

## Notes

- Only PDF files are accepted for upload, and text extraction depends on the PDF containing extractable text. Scanned/image-only PDFs need OCR, which is not included.
- Keep policy documents and credentials appropriate to the environment where the app runs. Uploaded PDFs are written to the configured documents directory.
- This project is a reference implementation, not a substitute for reviewing authoritative policies or obtaining professional advice.
