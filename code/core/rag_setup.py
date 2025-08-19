# rag_setup.py

# ==================================================================================================
# --- 1. Import Dependencies ---
# ==================================================================================================
# --- Standard Library Imports ---
import os
import logging
import subprocess
import json
import sys
from pathlib import Path

# --- Local Application Imports ---
from . import config

# --- Third-Party Library Imports ---
# LangChain is a framework for developing applications powered by language models.
# It provides abstractions for components like vector stores, LLMs, prompt templates, and chains.
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama allows for running large language models locally.
import ollama

# --- Imports for the Re-ranking Mechanism ---
# The re-ranking step refines the initial retrieval results for better quality.
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ==================================================================================================
# --- 2. Core Functions ---
# ==================================================================================================

def run_preprocessing_pipeline(cfg):
    """
    Executes an external Python script for document pre-processing.

    This function is responsible for the initial data preparation stage of the RAG pipeline.
    It invokes the `pre-processing.py` script, which handles tasks such as loading raw
    documents (PDFs, TXTs), splitting them into manageable chunks, and generating initial
    metadata. Using a separate script for this process promotes a modular architecture,
    allowing the pre-processing logic to be developed and tested independently.

    The function uses the `subprocess` module to ensure the script runs within the same
    Python environment, guaranteeing access to all necessary packages. It also captures
    the script's standard output and error streams for robust logging and debugging.
    """
    logging.info(f"Executing external pre-processing script: {cfg.PREPROCESSING_SCRIPT_NAME}...")
    try:
        # Ensure the directory for processed documents exists before running the script.
        os.makedirs(cfg.PROCESSED_DOC_FOLDER, exist_ok=True)

        # Use `sys.executable` to reference the current Python interpreter. This is a best
        # practice for ensuring that the subprocess uses the same environment and dependencies.
        python_executable = sys.executable

        # Construct the command to be executed, including arguments for input/output directories
        # and chunking parameters, which are sourced from the central config file.
        command = [
            python_executable, str(cfg.PREPROCESSING_SCRIPT_NAME),
            "--input_dir", str(cfg.RAW_DOC_FOLDER),
            "--output_dir", str(cfg.PROCESSED_DOC_FOLDER),
            "--chunk_size", str(cfg.CHUNK_SIZE),
            "--overlap_size", str(cfg.OVERLAP_SIZE),
        ]

        # The `subprocess.run` function executes the command.
        result = subprocess.run(
            command,
            check=True,          # Raises a `CalledProcessError` if the script returns a non-zero exit code (i.e., an error occurred).
            capture_output=True, # Captures stdout and stderr from the script.
            text=True,           # Decodes stdout and stderr as text.
            encoding="utf-8",    # Specifies UTF-8 encoding for cross-platform compatibility.
            cwd=str(cfg.BASE_DIR)
        )

        # Log the successful execution and any output from the script for transparency.
        logging.info("Pre-processing script completed successfully.")
        if result.stdout:
            logging.info(f"SCRIPT STDOUT: {result.stdout}")
        if result.stderr:
            logging.warning(f"SCRIPT STDERR: {result.stderr}")

    except FileNotFoundError:
        # This error occurs if the pre-processing script itself cannot be found.
        logging.error(f"Error: The script '{cfg.PREPROCESSING_SCRIPT_NAME}' was not found.")
        raise
    except subprocess.CalledProcessError as e:
        # This block catches errors that occur *during* the execution of the script.
        # It formats a detailed error message for easier diagnosis.
        error_message = (
            f"The pre-processing script failed with exit code {e.returncode}.\n"
            f"--- STDERR ---\n{e.stderr}\n"
            f"--- STDOUT ---\n{e.stdout}"
        )
        logging.error(error_message)
        # Re-raise the exception as a RuntimeError to halt the setup process,
        # as a failed pre-processing step is a critical failure.
        raise RuntimeError(error_message) from e

def init_ollama_model(cfg):
    """
    Initializes a custom Large Language Model (LLM) using the local Ollama service.

    This function ensures that the specific LLM required for the chatbot is available in Ollama.
    It first queries Ollama for a list of existing models. If the target model (defined in `config.py`)
    is not found, it creates a new custom model. This custom model is based on a foundational model
    (e.g., 'gemma3:4b') but is configured with a specific system prompt and parameters.

    The system prompt is critical for defining the chatbot's persona, instructions, and constraints,
    guiding its behavior during conversation.
    """
    logging.info("Initializing Ollama model...")
    try:
        # Retrieve the list of all models currently available in the local Ollama instance.
        available_models = ollama.list()
        
        # Safely extract model names. The response from ollama.list() might contain entries
        # without a 'name' key, so we iterate carefully.
        model_names = []
        for model in available_models.get('models', []):
            if 'name' in model:
                model_names.append(model['name'])

        # Check if the desired custom model already exists.
        if cfg.MODEL_NAME not in model_names:
            logging.info(f"Model '{cfg.MODEL_NAME}' not found. Creating it now...")
            # If the model does not exist, create it using `ollama.create`.
            ollama.create(
                model=cfg.MODEL_NAME,
                from_="gemma3:4b",  # Specifies the base model to build upon.
                # `parameters` allows for tuning model behavior. Temperature controls randomness.
                parameters={"temperature": 0.5},
                # The `system` parameter sets a persistent system prompt that defines the model's role and instructions.
                # This is a key part of prompt engineering for reliable behavior.
                system="""You are a helpful university chatbot for the course 'Machine Learning for Management Decisions'
                at Goethe University Frankfurt.
                - Your answers must be brief, concise, and based on the provided documents.
                - You can understand and connect questions and documents even if they are in different languages.
                - CRITICALLY, you MUST answer in the same language as the user's question. If the user asks in German, respond in German.
                If the user asks in English, respond in English."""
            )
            logging.info(f"Custom model '{cfg.MODEL_NAME}' created successfully.")
        else:
            logging.info(f"Custom model '{cfg.MODEL_NAME}' already exists.")
    except Exception as e:
        logging.error(f"Failed to initialize Ollama model: {e}")
        raise

def initialize_and_update_vector_db(embeddings: HuggingFaceEmbeddings, cfg) -> Chroma:
    """
    Initializes or loads a Chroma vector database and updates it with new documents.

    This function manages the persistence and updating of the vector store, which is the core
    of the "Retrieval" in RAG. It maintains a log file (`ingested_files.log`) to track which
    documents have already been processed and vectorized.

    On execution, it compares the files in the processed documents folder with the log to
    identify new files. These new files are then loaded, their chunks are converted into
    LangChain `Document` objects, and they are added to the Chroma database. This incremental
    updating process is efficient as it avoids re-processing the entire corpus each time.

    Args:
        embeddings: The initialized HuggingFace embedding model used to convert text to vectors.

    Returns:
        An instance of the Chroma vector database, ready for querying.
    """
    # The manifest file acts as a ledger of processed files to prevent redundant ingestion.
    manifest_path = cfg.PERSIST_DIRECTORY / "ingested_files.log"
    os.makedirs(cfg.PERSIST_DIRECTORY, exist_ok=True)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            ingested_files = set(f.read().splitlines())
    except FileNotFoundError:
        # If the log file doesn't exist, it means no files have been ingested yet.
        ingested_files = set()

    # Identify all available processed JSON files.
    all_processed_files = set()
    if os.path.exists(cfg.PROCESSED_DOC_FOLDER):
        for filename in os.listdir(cfg.PROCESSED_DOC_FOLDER):
            if filename.startswith("chunks_") and filename.endswith(".json"):
                all_processed_files.add(os.path.join(cfg.PROCESSED_DOC_FOLDER, filename))

    # Determine which files are new by calculating the set difference.
    new_files_to_add = sorted(list(all_processed_files - ingested_files))

    # Initialize the Chroma vector store. If a database already exists at the `persist_directory`,
    # it will be loaded; otherwise, a new one will be created.
    vector_db = Chroma(
        persist_directory=str(cfg.PERSIST_DIRECTORY),
        embedding_function=embeddings,
        collection_name=cfg.VECTOR_STORE_NAME
    )

    if not new_files_to_add:
        logging.info("Vector database is up-to-date. No new documents to ingest.")
        return vector_db

    logging.info(f"Found {len(new_files_to_add)} new document(s) to add to the vector database.")

    # Process each new file.
    for file_path in new_files_to_add:
        logging.info(f"Ingesting document: {os.path.basename(file_path)}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents_to_add = []
                for chunk in data.get("chunks", []):
                    # ChromaDB requires metadata values to be simple types (str, int, float).
                    # This section sanitizes complex metadata (like lists of keywords) into strings.
                    metadata = chunk.get("metadata", {})
                    if "keywords" in metadata and isinstance(metadata["keywords"], list):
                        metadata["keywords"] = ", ".join(metadata["keywords"])
                    if "entities" in metadata and isinstance(metadata["entities"], list):
                        metadata["entities"] = json.dumps(metadata["entities"])

                    # Add the original source document name to each chunk's metadata for traceability.
                    metadata["source"] = data.get("document_source", os.path.basename(file_path))

                    # Create a LangChain `Document` object, which is the standard format for text chunks.
                    doc = Document(
                        page_content=chunk.get("text", ""),
                        metadata=metadata
                    )
                    documents_to_add.append(doc)

            # Add the batch of documents from the file to the vector database.
            if documents_to_add:
                vector_db.add_documents(documents_to_add)
                logging.info(f"Successfully added {len(documents_to_add)} chunks from {os.path.basename(file_path)}.")
                # After successful ingestion, update the manifest log.
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(f"{file_path}\n")
        except Exception as e:
            logging.error(f"Failed to process and ingest {file_path}: {e}")

    return vector_db

def create_chain(retriever, llm, memory):
    """
    Constructs the final ConversationalRetrievalChain.

    This function assembles the core components of the RAG system into a single, executable
    LangChain chain. It defines a custom prompt template (`CONDENSE_QUESTION_PROMPT`) that is
    crucial for handling conversational context. This prompt instructs the LLM on how to
    rephrase a follow-up question into a standalone question by using the chat history.
    For example, if the history is "What is RAG?" and the user asks "How does it work?", the
    LLM will generate a new, self-contained question like "How does RAG work?" for the retrieval step.

    Args:
        retriever: The configured document retriever (in this case, a re-ranking retriever).
        llm: The initialized language model.
        memory: The conversation memory buffer.

    Returns:
        A fully configured `ConversationalRetrievalChain` instance.
    """
    logging.info("Creating the conversational retrieval chain...")

    # This prompt template is vital for making the RAG system conversational.
    # It guides the LLM to synthesize chat history and a new question into a
    # standalone query suitable for vector retrieval.
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    **Crucially, the standalone question MUST be in the exact same language as the "Follow Up Input".**

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # The `ConversationalRetrievalChain` orchestrates the entire RAG process:
    # 1. It takes the chat history and the new question.
    # 2. It uses the `condense_question_prompt` and the LLM to create a standalone question.
    # 3. It passes this standalone question to the `retriever`.
    # 4. The `retriever` fetches relevant documents.
    # 5. It passes the documents and the original question to the LLM to generate a final answer.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,  # Ensures that the source documents are returned with the answer.
        verbose=False                  # Set to True for detailed, step-by-step logging of the chain's execution.
    )
    logging.info("Conversational chain created successfully.")
    return chain

def setup_chatbot(cfg):
    """
    Main orchestrator function to initialize and assemble the entire RAG chatbot system.

    This function serves as the entry point for building the RAG pipeline. It executes all
    necessary setup steps in a sequential and logical order, from data pre-processing to
    the final chain assembly.

    Returns:
        A tuple containing the fully configured RAG chain, the conversation memory object,
        and the vector database instance. These components are then used by the application
        layer (e.g., a Streamlit UI) to power the chatbot.
    """
    logging.info("--- Starting Full RAG Chatbot System Setup ---")

    # Step 1: Execute the data pre-processing pipeline.
    run_preprocessing_pipeline(cfg)

    # Step 2: Initialize the custom Ollama LLM.
    init_ollama_model(cfg)

    # Step 3: Initialize the sentence-transformer model for creating text embeddings.
    # This model runs locally on the specified device ('cpu' or 'cuda').
    logging.info(f"Initializing embedding model: {cfg.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use "cuda" for GPU acceleration.
        encode_kwargs={"normalize_embeddings": False} # Normalization can be useful but is not always necessary.
    )

    # Step 4: Initialize the vector database, loading existing data and ingesting new documents.
    vector_db = initialize_and_update_vector_db(embeddings, cfg)
    if not vector_db:
        # This is a critical failure point; the system cannot function without a vector DB.
        logging.error("Vector DB initialization failed. Aborting setup.")
        raise RuntimeError("Failed to initialize the vector database.")

    # Step 5: Initialize the LangChain wrapper for the Ollama LLM.
    llm = ChatOllama(model=cfg.MODEL_NAME)

    # Step 6: Create the base retriever from the vector database.
    # This retriever performs the initial, broad search for relevant documents.
    base_retriever = vector_db.as_retriever(
        search_kwargs={"k": cfg.INITIAL_RETRIEVAL_K} # Retrieves the top K documents.
    )

    # Step 7: Initialize the re-ranker model. A cross-encoder is used here, which is more
    # computationally expensive but more accurate than the initial vector search.
    logging.info(f"Initializing re-ranker model: {cfg.RERANKER_MODEL}")
    cross_encoder_model = HuggingFaceCrossEncoder(model_name=cfg.RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=cfg.TOP_N_RERANKED)

    # Step 8: Create the `ContextualCompressionRetriever`. This is a two-stage retrieval process.
    # The `base_retriever` first fetches a larger set of documents (K=6).
    # The `compressor` (the re-ranker) then re-scores and filters this set down to the
    # most relevant documents (top_n=3), improving the quality of the context provided to the LLM.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    logging.info(f"Re-ranking retriever created. Will retrieve {cfg.INITIAL_RETRIEVAL_K} docs and keep the top {cfg.TOP_N_RERANKED}.")

    # Step 9: Initialize the conversation memory. `ConversationBufferWindowMemory` stores
    # the last `k` interactions, providing short-term context for follow-up questions.
    memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="chat_history", # The key used in the prompt template.
        output_key="answer"       # The key for the final answer in the chain's output.
    )

    # Step 10: Assemble all the components into the final conversational chain.
    chain = create_chain(compression_retriever, llm, memory)

    logging.info("--- RAG Chatbot Setup Complete ---")
    return chain, memory, vector_db
