# rag_utils.py

# ==================================================================================================
# --- 1. Import Dependencies ---
# ==================================================================================================
import ollama
import re
import streamlit as st
# **FIX**: Use a relative import to access the config file from within the same package.
from . import config

# ==================================================================================================
# --- 2. Utility Functions ---
# ==================================================================================================

def get_lecture_filter_from_query(question: str) -> int:
    """
    Uses an LLM to dynamically analyze a user's query for specific filter criteria.

    This function implements a key feature of the dynamic RAG pipeline: query-driven
    retrieval filtering. It sends the user's question to a smaller, faster LLM with a
    prompt engineered to extract a specific piece of information, in this case, a lecture
    number. This allows the system to narrow down the search space in the vector store
    before retrieval, leading to more relevant results.

    Args:
        question: The user's input string.

    Returns:
        The lecture number as an integer if found; otherwise, returns 0.
    """
    try:
        # The prompt is carefully structured to instruct the LLM on its specific task:
        # identify a lecture number and return ONLY that number or 0.
        prompt = f"""
        Analyze the following question to identify if it refers to a specific lecture number.
        - If a number is written as a word (e.g., "first", "second", "fifth"), convert it to its digit form (e.g., 1, 2, 5).
        - Your response must be ONLY the integer number of the lecture.
        - If no specific lecture number is mentioned, your response must be ONLY the number 0.

        Question: "{question}"
        """

        # Initialize the Ollama client to communicate with the local LLM.
        client = ollama.Client()
        response = client.chat(
            model=config.OLLAMA_METADATA_MODEL, 
            messages=[{'role': 'user', 'content': prompt}],
        )

        # Parse the LLM's response to extract the numerical value.
        content = response['message']['content'].strip()
        match = re.search(r'\d+', content)
        if match:
            return int(match.group(0))

    except Exception as e:
        # If the LLM call fails, log a warning to the UI but do not crash the application.
        st.warning(f"Could not determine lecture filter from query: {e}")

    # Default to 0 if no lecture number is found or if an error occurs.
    return 0
