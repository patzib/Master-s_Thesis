# RAG Chatbot for University Course Q&A

## 1. Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about the university course "Machine Learning for Management Decisions." The system leverages local large language models (LLMs) via Ollama and a vector database to provide accurate, context-aware answers based on a corpus of course documents, including lecture slides, transcripts, and academic papers.

Key features include a two-stage retrieval process with re-ranking, dynamic metadata filtering based on query analysis, and conversational memory to handle follow-up questions. The entire application is served through an interactive web interface built with Streamlit.

## 2. System Architecture

The system is composed of several modular Python scripts that handle distinct stages of the RAG pipeline:

    pre-processing.py: This script is the first stage of the data pipeline. It reads raw documents (PDFs, TXTs) from the data/raw_documents folder, splits them into manageable text chunks, and enriches them with LLM-generated metadata (titles, summaries, keywords). The structured output is saved as JSON files in data/processed_output.

    rag_setup.py: This is the core setup script that orchestrates the creation of the RAG chain. It runs the pre-processing pipeline, initializes the local Ollama LLM, creates or loads the Chroma vector database with the processed documents, and assembles the final ConversationalRetrievalChain.

    config.py: A centralized configuration file that stores all important parameters, such as file paths, model names, and RAG hyperparameters (e.g., chunk size, retrieval K). This allows for easy tuning and experimentation.

    rag_utils.py: Contains helper functions used by the main application. Its primary role is to analyze the user's query in real-time to extract potential metadata filters (like a specific lecture number).

    app.py: The main application file that runs the Streamlit web interface. It initializes the RAG system on startup, manages the session state (chat history), and handles the user interaction loop.

The data flows as follows:
Raw Documents -> pre-processing.py -> JSON Chunks -> rag_setup.py -> Chroma Vector DB -> app.py -> User Interface

## 3. Features

    End-to-End RAG Pipeline: Implements the full RAG lifecycle from document ingestion to conversational response generation.

    Local LLMs: Utilizes Ollama to run language models locally, ensuring data privacy and cost-free operation.

    Advanced Retrieval: Employs a two-stage retrieval process with a CrossEncoderReranker to improve the relevance of documents passed to the LLM.

    Dynamic Metadata Filtering: Intelligently analyzes user queries to apply filters to the vector search, leading to more precise context retrieval.

    Conversational Memory: Maintains a history of the conversation to accurately answer follow-up questions.

    Interactive UI: Provides a user-friendly chat interface powered by Streamlit, complete with source document display.

## 4. Setup and Installation

Follow these steps to set up and run the project locally.
Prerequisites

    Python 3.9+

    Ollama installed and running.

    The required Ollama models pulled. You can get them by running:

    Example: ollama pull gemma3:4b

Installation Steps

    Clone the Repository:

    git clone https://github.com/patzib/Master-s_Thesis
    cd Master-s_Thesis

    Create a Virtual Environment (Recommended):
    python -m venv venv # If that does not work: py -3 -m venv venv
    
    Activate Virtual Environment;
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate.bat`

    Navigate to the MLME_chatbot folder
    # Or wherever the requirements.txt is saved

    Install Dependencies:
    pip install -r requirements.txt

    Set Up Folder Structure:
    Create the necessary data directory in the project's root folder:

    mkdir -p data/raw_documents

    Place all your source documents (PDFs, TXTs) into the data/raw_documents folder.

## 5. Usage

Once the setup is complete, you can run the Streamlit application:

streamlit run app.py

The application will open in your web browser. The first time you run it, the system will perform the initial pre-processing and database indexing, which may take a few minutes depending on the number of documents. Subsequent startups will be much faster as the indexed data is persisted.

## 6. Configuration

All key parameters of the system can be modified in the config.py file. This allows for easy experimentation with different models, chunking strategies, and retrieval settings.

    Paths: RAW_DOC_FOLDER, PROCESSED_DOC_FOLDER, PERSIST_DIRECTORY

    Models: MODEL_NAME, EMBEDDING_MODEL, RERANKER_MODEL

    RAG Parameters: CHUNK_SIZE, OVERLAP_SIZE, INITIAL_RETRIEVAL_K, TOP_N_RERANKED

By tuning these parameters, you can analyze their effect on the performance and accuracy of the RAG system.
