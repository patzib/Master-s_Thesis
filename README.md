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

    Navigate to the "Master-s_Thesis" folder
    cd Master-s_Thesis

    Create a Virtual Environment (Recommended):
    python -m venv venv # If that does not work: py -3 -m venv venv
    
    Activate Virtual Environment:
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate.bat`

    Install Dependencies:
    pip install -r requirements.txt

    Delete the placeholder in the data/raw_documents and place all your source documents (only PDF and TXT files) into that folder.
For the lecture filter function to work correctly, the documents that are from a specific lecture need to be renamed accordingly.
For example: lecture_01.pdf or lecture_03_transcript.txt



## 5. Usage

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

Before running the evaluations, it is recommended to run and initialize the chatbot once before starting directly with the evaluations (streamlit run app.py). That way, the pre-processing of the files has already been done.

To start the evaluations, install the necessary packages:

    pip install -r requirements_eval.txt

When the installations have finished, the notebooks for evaluating the RAG system can be run with jupyter notebook. For starting jupyter notebook, enter:

    jupyter notebook

If you would like to configue the chatbot with certain settings, then you need to go first to the config.py file in the core folder. There parameters like the temperature or the System Prompt can be adjusted.

Then, navigate to the code/evaluation folder.

### 7.1 Evaluating the RAG chatbot on synthetically created and manually curated Question-Answer Pairs.

The next step would be to run the q_a_generator.ipynb notebook, prompt an LLM of your choice for question-answer pairs for the relevant academic papers and then manually bringing them together, forming the ground_truth_dataset.csv, with which the model is then later tested.

Alternatively, this csv file is already created and saved in the evaluation_data folder.

Then, run the rag_eval.ipynb script for generating the answers of the chatbot to the questions present in the ground truth dataset. There already is one example of this dataset in the evaluation_data folder for model V4.

After that, take an LLM of your choice, give it the created csv from the step before (rag_evaluation_generated_answers_{model_version}.csv) and use the prompts which you find in the Appendix of the Thesis to evaluate the Chatbot on the metrics of Faithfulness, Context Relevance and Correctness.

The answers can then be copied into three distinct json files (faithfulness_{model_version}.json, correctness_{model_version}.json, relevance_{model_version}.json). Using the script convert_json_to_csv.ipynb then converts all three json files into one single csv file, containing all the information of the evaluation of this one model version (consolidated_evaluation_results_custom_dataset_{model_version}.csv). 

There are already three example json files for this in the evaluation_data folder which were created with the following settings of the chatbot: Temperature = 0.1 and Language = English Only (V4).


### 7.2 Evaluating the RAG chatbot on established benchmarks

In this step, the chatbot is evaluated against its base model to see how well they perform on established benchmarks, namely TruthfulQA and TriviaQA.

First, run the rag_eval_benchmark_tests.ipynb notebook to create the two csv's which contain the answers of the chatbot and its base model on the TriviaQA and TruthfulQA datasets. 

These csv's are then attached to an LLM of your choice and evaluated using the prompts which can be found in the Thesis. The output of the LLM can then be saved in two distinct csv files. 

