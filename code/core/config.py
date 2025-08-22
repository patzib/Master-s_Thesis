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
MODEL_NAME = "MLME_chatbot_v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
OLLAMA_METADATA_MODEL = "gemma3:4b"
VECTOR_STORE_NAME = "rag_store_MLME"

# --- RAG Configuration ---
CHUNK_SIZE = 1250
OVERLAP_SIZE = 250
INITIAL_RETRIEVAL_K = 6
TOP_N_RERANKED = 3
METADATA_GENERATION_CHAR_LIMIT = 40000