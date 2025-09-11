"""
pre-processing.py

This script is responsible for the data ingestion and preparation pipeline.
It processes raw source documents (.pdf, .txt), enriches them with
LLM-generated metadata (titles, summaries, keywords), splits them into
semantically coherent chunks, and saves the structured output as JSON files,
ready for ingestion into the vector database.
"""

# ==================================================================================================
# --- 1. Import Dependencies ---
# ==================================================================================================
import os
import json
import argparse
import ollama
import hashlib
from datetime import datetime
import re
import logging
import sys      
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import config 

# LangChain components for document loading and text splitting.
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==================================================================================================
# --- 2. Logging Configuration ---
# ==================================================================================================
# Standard logging setup to provide visibility into the script's execution.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ==================================================================================================
# --- 3. Core Pre-processing Functions ---
# ==================================================================================================

def generate_metadata_with_llm(client: ollama.Client, text_content: str, filename: str) -> dict:
    """
    Uses a Large Language Model (LLM) for automated metadata enrichment.

    This function sends the initial text content of a document to an LLM to generate
    structured metadata, including a title, summary, and keywords. This automated
    enrichment provides valuable, high-level context for each document that can be
    leveraged during the retrieval process.

    Args:
        client: The Ollama client instance.
        text_content: The raw text content of the document.
        filename: The original filename, used for context in the prompt.

    Returns:
        A dictionary containing the LLM-generated metadata.
    """
    logging.info(f"Generating LLM-based metadata for {filename}...")
    # This prompt is engineered to instruct the LLM to return a valid JSON object.
    prompt = f"""
    Based on the following text from the document '{filename}', please generate the following metadata:
    1.  "title": A short, descriptive title for the document.
    2.  "summary": A concise 2-3 sentence summary of the main points.
    3.  "keywords": A list of 5-7 important keywords or topics.
    Respond ONLY with a valid JSON object. Do not include any other text or explanations.
    Text content:
    ---
    {text_content[:config.METADATA_GENERATION_CHAR_LIMIT]} # Limit the text sent to the LLM.
    ---
    """
    try:
        response = client.chat(
            model=config.OLLAMA_METADATA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            format="json"  # Instructs Ollama to ensure the output is valid JSON.
        )
        metadata = json.loads(response['message']['content'])
        logging.info(f"Successfully generated metadata for {filename}.")
        return metadata
    except Exception as e:
        logging.warning(f"Could not generate AI metadata for {filename}. Error: {e}")
        # Return a default structure in case of failure.
        return {"title": f"Title for {filename}", "summary": "Summary not available.", "keywords": []}

def process_document(filepath: str, chunk_size: int, chunk_overlap: int, client: ollama.Client) -> dict | None:
    """
    Loads, chunks, and structures a single source document.

    This is the central function of the pre-processing script. It performs the following steps:
    1. Loads a document (PDF or TXT) into memory.
    2. Extracts metadata (e.g., lecture number) from the filename.
    3. Calls `generate_metadata_with_llm` for automated metadata enrichment.
    4. Splits the document's text into smaller, overlapping chunks.
    5. Structures the chunks and metadata into a single JSON object for later ingestion.

    Args:
        filepath: The path to the source document.
        chunk_size: The target size for each text chunk.
        chunk_overlap: The overlap size between adjacent chunks.
        client: The Ollama client instance.

    Returns:
        A dictionary containing the structured data for the document, or None if processing fails.
    """
    base_name = os.path.basename(filepath)
    logging.info(f"--- Starting processing for: {base_name} ---")

    # Use regex to extract a lecture number from the filename if present.
    match = re.search(r'\d+', base_name)
    lecture_number = int(match.group(0)) if match else 0

    # Select the appropriate LangChain document loader based on the file extension.
    if filepath.lower().endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif filepath.lower().endswith('.txt'):
        loader = TextLoader(filepath, encoding='utf-8')
    else:
        logging.warning(f"Unsupported file type: {base_name}. Skipping.")
        return None

    documents = loader.load()
    full_text = " ".join([doc.page_content for doc in documents])

    if not full_text.strip():
        logging.warning(f"File is empty: {base_name}. Skipping.")
        return None

    # Generate high-level metadata using the LLM.
    llm_metadata = generate_metadata_with_llm(client, full_text, base_name)

    # Initialize the text splitter with parameters from the config file.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split {base_name} into {len(chunks)} chunks.")

    # Structure the final output, combining document-level and chunk-level metadata.
    output_chunks = []
    for i, chunk in enumerate(chunks):
        final_metadata = {
            "document_title": llm_metadata.get("title", base_name),
            "document_summary": llm_metadata.get("summary", ""),
            "document_keywords": ", ".join(llm_metadata.get("keywords", [])),
            "source_document": base_name,
            "lecture_number": lecture_number,
            "chunk_position": i,
        }
        output_chunks.append({"text": chunk.page_content, "metadata": final_metadata})

    structured_data = {
        "document_source": base_name,
        "processed_at": datetime.now().isoformat(),
        "chunking_config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "llm_metadata": llm_metadata,
        "chunks": output_chunks
    }
    logging.info(f"--- Finished processing for: {base_name} ---")
    return structured_data

# ==================================================================================================
# --- 4. Main Execution Block ---
# ==================================================================================================
if __name__ == "__main__":
    # `argparse` is used to make the script executable from the command line,
    # allowing for flexible configuration of input/output directories and chunking parameters.
    parser = argparse.ArgumentParser(description="Process text and PDF documents for RAG.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for processed JSON files.")
    parser.add_argument("--chunk_size", type=int, default=1250, help="Target chunk size.")
    parser.add_argument("--overlap_size", type=int, default=250, help="Overlap between chunks.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify connection to the Ollama service before starting processing.
    try:
        ollama_client = ollama.Client()
        ollama_client.show(config.OLLAMA_METADATA_MODEL)
        logging.info(f"Successfully connected to Ollama and found model '{config.OLLAMA_METADATA_MODEL}'.")
    except Exception as e:
        logging.error(f"Error connecting to Ollama. Please ensure it is running. Details: {e}")
        exit()

    # Iterate over all supported files in the input directory.
    files_to_process = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.txt', '.pdf'))]

    for filename in files_to_process:
        input_filepath = os.path.join(args.input_dir, filename)
        base_name_no_ext = os.path.splitext(filename)[0]
        output_filepath = os.path.join(args.output_dir, f'chunks_{base_name_no_ext}.json')

        # Skip files that have already been processed to avoid redundant work.
        if os.path.exists(output_filepath):
            logging.info(f"Output for '{filename}' already exists. Skipping.")
            continue

        try:
            processed_data = process_document(
                filepath=input_filepath,
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap_size,
                client=ollama_client
            )
            # Save the structured data to a JSON file.
            if processed_data:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                logging.info(f"Successfully saved processed data to '{output_filepath}'\n")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {filename}: {e}\n")