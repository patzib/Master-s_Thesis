"""
config.py

This script centralizes all key parameters and configurations for the RAG
application. By defining paths, model identifiers, and RAG settings in one
place, it allows for easy modification and tuning of the system without
altering the core application logic.
"""

from pathlib import Path

# --- Path Configuration ---
# Root of the Python application (the "Master's Thesis" folder)
BASE_DIR = Path(__file__).parent.parent
# PROJECT_ROOT is the top-level folder ('code')
PROJECT_ROOT = BASE_DIR.parent

RAW_DOC_FOLDER = PROJECT_ROOT / "data" / "raw_documents"
PROCESSED_DOC_FOLDER = PROJECT_ROOT / "data" / "processed_output"
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
PREPROCESSING_SCRIPT_NAME = BASE_DIR / "core" / "pre-processing.py"

# --- Model Configuration ---
MODEL_NAME = "MLME_chatbot_v4"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
OLLAMA_BASE_MODEL = "gemma3:4b"
OLLAMA_METADATA_MODEL = "gemma3:4b"
VECTOR_STORE_NAME = "rag_store"

# --- LLM Behavior Configuration ---
SYSTEM_PROMPT = """You are a helpful university chatbot for the course 'Machine Learning for Management Decisions'
at Goethe University Frankfurt.
- Your answers must be brief, concise, and based on the provided documents.
- You can understand and connect questions and documents even if they are in different languages.
- CRITICALLY, you MUST answer ONLY in English, regardless of the language of the user's question."""
# For the Multilingual System Prompt, exchange the last line with: “CRITICALLY, you MUST answer in the same language 
# as the user's question. If the user asks in German, respond in German. If the user asks in English, respond in English.

# Controls the creativity of the LLM. Lower values are more deterministic.
OLLAMA_TEMPERATURE = 0.1

# --- RAG Configuration ---
CHUNK_SIZE = 1250
OVERLAP_SIZE = 250
INITIAL_RETRIEVAL_K = 6
TOP_N_RERANKED = 3
METADATA_GENERATION_CHAR_LIMIT = 40000

