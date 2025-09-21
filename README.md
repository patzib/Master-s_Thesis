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

The data pipeline follows this logical flow:

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

    Navigate to the "Master-s_Thesis" folder
    cd Master-s_Thesis

    Create a Virtual Environment (Recommended):
    python -m venv venv # If that does not work: py -3 -m venv venv
    
    Activate Virtual Environment:
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate.bat`

    Install Dependencies:
    pip install -r requirements.txt

Data Preparation

Delete the placeholder in the data/raw_documents and place all your source documents (only PDF and TXT files) into that folder.
Note on File Naming: For the metadata filter to function correctly, files must be named to reflect their corresponding lecture number (e.g., lecture_01.pdf, lecture_03_transcript.txt).    

## 5. Execution

Once the setup is complete, navigate to the "code" folder and run the Streamlit application:

    cd code
    
    streamlit run app.py

The application will open in your web browser. The first time you run it, the system will perform the initial pre-processing and database indexing, which may take a few minutes depending on the number of documents in the raw_documents folder. Subsequent startups will be faster as the indexed data is persisted.

## 6. Configuration

All key parameters of the system can be modified in the config.py file. This allows for easy experimentation with different models, chunking strategies, and retrieval settings.

    Paths: RAW_DOC_FOLDER, PROCESSED_DOC_FOLDER, PERSIST_DIRECTORY

    Models: MODEL_NAME, EMBEDDING_MODEL, RERANKER_MODEL, etc.

    RAG Parameters: CHUNK_SIZE, OVERLAP_SIZE, INITIAL_RETRIEVAL_K, TOP_N_RERANKED, METADATA_GENERATION_CHAR_LIMIT, OLLAMA_TEMPERTAURE, SYSTEM_PROMPT

By tuning these parameters, you can analyze their effect on the performance and accuracy of the RAG system.


## 7. Running the Evaluations

It is advisable to run the main application (streamlit run app.py) at least once prior to initiating evaluations to ensure all data pre-processing is complete.
The project includes sample files to demonstrate each step. However, it is crucial to note that the rag_eval script generates new output files with each run. Therefore, after running this script, the original sample .json files in the evaluation_data folder will be outdated. Then, newly generated JSON files must be used for the convert_json_to_csv script to ensure data consistency.

To start the evaluations, install the necessary packages:

    pip install -r requirements_eval.txt

When the installations have finished, the notebooks for evaluating the RAG system can be run with jupyter notebook. For starting jupyter notebook, enter:

    jupyter notebook

If you would like to evaluate the chatbot with certain settings, then you need to go first to the config.py file in the core folder. There parameters like the temperature or the System Prompt can be adjusted.

Within the jupyter notebook interface, navigate to the code/evaluation folder.

### 7.1 Evaluation on Custom Dataset

This procedure assesses the system's performance on a synthetically generated question-answer dataset derived from the source documents. For every step, exemplary files are saved in the corresponding folders.

1. Generate Q&A Pairs: Execute the q_a_generator.ipynb notebook to generate question-answer pairs from the document corpus. These should be manually curated to create a ground_truth_dataset.csv. An example of this file (with which the answers were generated in the thesis) is provided in the evaluation_data directory. The csv file also has five LLM-generated (from Gemini 2.5 Pro) question-answer pairs for each academical papper that was in the source documents. This is optional but increases the quality of the data.
2. Generate Model Responses: Run rag_eval.ipynb to have the system generate answers to the questions in the ground truth dataset. This will produce a file named rag_evaluation_generated_answers_{model_version}.csv. Again, there is an example file already saved in the evaluation_data folder.
3. Evaluate via LLM-as-a-Judge: Employ a powerful LLM (e.g., ChatGPT or Gemini) and the prompts specified in the thesis appendix to evaluate the generated answers on the metrics of Faithfulness, Context Relevance, and Correctness. In the evaluation_data folder, exemplary json files are deposited, with which the next evaluation step can be made.
4. Consolidate Results: The output from the LLM judge, typically saved as three JSON files, can be merged into a single CSV file for analysis using the convert_json_to_csv.ipynb script.

### 7.2 Evaluation on Standard Benchmarks

This procedure benchmarks the RAG system against its base model using the TruthfulQA and TriviaQA datasets.

1. Generate Benchmark Responses: Run the rag_eval_benchmark_tests.ipynb notebook to generate two CSV files containing answers from both the RAG system and its base LLM for the benchmark questions.
2. Evaluate via LLM-as-a-Judge: Utilize an external LLM and the evaluation prompts from the thesis to score the outputs from both models. The results can be saved into new CSV files for analysis.
