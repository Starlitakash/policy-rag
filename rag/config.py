import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage paths
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "documents"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sritr_university_policies")

# Qdrant Configuration
# Option 1: Qdrant Cloud / Remote Server (e.g. https://xxx.cloud.qdrant.io)
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

# Option 2: Local Persistent File Storage (Used if QDRANT_URL is not set)
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", str(BASE_DIR / "qdrant_db")).strip()

# Ensure directories exist
Path(DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
if not QDRANT_URL and QDRANT_STORAGE_PATH:
    Path(QDRANT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval parameters
TOP_K = int(os.getenv("TOP_K", "4"))

# Embeddings: FastEmbed model (384 dimensions)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION = 384

# Sole LLM Provider: Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

# Server Settings
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
API_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
