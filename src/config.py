"""Configuration management for Plant Care AI."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file if it exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Retrieval Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.25"))
DEFAULT_RETRIEVAL_MODE = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid")

# Directory paths
KNOWLEDGE_DIR = BASE_DIR / os.getenv("KNOWLEDGE_DIR", "knowledge")
CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", ".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings_cache.pkl"
