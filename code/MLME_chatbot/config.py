from pathlib import Path

# --- Path Configuration ---
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DOC_FOLDER = BASE_DIR / "data" / "raw_documents"
PROCESSED_DOC_FOLDER = BASE_DIR / "data" / "processed_output"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
PREPROCESSING_SCRIPT_NAME = "pre-processing.py"

# --- Model Configuration ---
MODEL_NAME = "MLME_chatbot"
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