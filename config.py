from pathlib import Path

from dotenv import load_dotenv

# Project root; all paths are based on it, so the working directory doesn't matter
BASE_DIR = Path(__file__).parent

# Load LangSmith settings from .env to enable tracing
load_dotenv(BASE_DIR / ".env")

# Models
CHAT_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"

# RAG: medical guideline document and vector store location
GUIDELINE_PATH = BASE_DIR / "data" / "medical_guideline.md"
INDEX_PATH = BASE_DIR / "faiss_index_medical"

# MCP Server: uses port 8001 to avoid clashing with other local servers on 8000
MCP_HOST = "127.0.0.1"
MCP_PORT = 8001
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
