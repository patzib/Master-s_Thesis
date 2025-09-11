"""
app.py

This script serves as the main entry point for the RAG chatbot application.
It uses Streamlit to create a web-based user interface where users can
interact with the chatbot.

The script handles the following key responsibilities:
- Initializing the entire RAG pipeline on the first run using the setup_chatbot function.
- Caching the initialized components in the Streamlit session state to avoid costly reloads.
- Managing the chat interface, including displaying conversation history and sources.
- Processing user queries in real-time, including dynamic metadata filtering.
"""

# ==================================================================================================
# --- 1. Import Dependencies ---
# ==================================================================================================
# --- Standard Library Imports ---
import os
from collections import defaultdict

# --- Third-Party Library Imports ---
import streamlit as st
from langchain_core.messages import HumanMessage

# --- Local Application Imports  ---

from core.rag_setup import setup_chatbot
from core import config as app_config
from core.rag_utils import get_lecture_filter_from_query

# ==================================================================================================
# --- 2. Core Application Logic ---
# ==================================================================================================

# In app.py

def query_with_memory(chain, question: str, vector_db):
    """
    Executes a query against the RAG chain with dynamic metadata filtering.

    This function intercepts the user's query to analyze it for specific filter
    criteria (e.g., a lecture number). If a filter is detected, it temporarily
    configures the chain's retriever to use that filter for the current query,
    leading to more precise and relevant document retrieval.

    Args:
        chain: The initialized ConversationalRetrievalChain object.
        question: The user's input question.
        vector_db: The Chroma vector database instance.

    Returns:
        A tuple containing the generated answer (str) and a list of source documents.
    """
    # Store the original base retriever to restore it later
    original_base_retriever = chain.retriever.base_retriever

    try:
        # Step 1: Analyze the user's query to detect if a lecture filter is needed.
        lecture_filter = get_lecture_filter_from_query(question)

        # Step 2: If a filter is found, create a new base_retriever and temporarily assign it.
        if lecture_filter > 0:
            st.write(f"Filtering search for Lecture {lecture_filter}...")
            # Create a new retriever with the filter
            filtered_retriever = vector_db.as_retriever(
                search_kwargs={"k": app_config.INITIAL_RETRIEVAL_K, "filter": {"lecture_number": lecture_filter}}
            )
            # Assign it to the 'base_retriever' attribute of the existing compression retriever
            chain.retriever.base_retriever = filtered_retriever

        # Step 3: Invoke the chain with the user's question to get the result.
        # The chain will now use the original (or the newly filtered) base retriever
        # and then apply the re-ranking step.
        result = chain.invoke({"question": question})
        answer = result.get("answer", "Sorry, I could not find an answer.")
        sources = result.get("source_documents", [])

        return answer, sources

    except Exception as e:
        st.error(f"An error occurred during the query process: {e}")
        return "I encountered an error while trying to answer your question. Please try again.", []
    finally:
        # Step 4: Always restore the original base retriever.
        # This ensures the next query doesn't accidentally use the filter from this query.
        chain.retriever.base_retriever = original_base_retriever

def main():
    """The main function that defines and runs the Streamlit application UI."""
    st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

    st.title("MLME Chatbot")
    st.markdown("Ask questions about the course 'Machine Learning for Management Decisions'!")

    # --- Automatic RAG System Initialization ---
    # This block runs ONCE per session. If the 'chain' is not in the session state,
    # it initializes the entire RAG pipeline and stores the components.
    if 'chain' not in st.session_state:
        st.session_state.chain = None
        st.session_state.memory = None
        st.session_state.messages = []
        st.session_state.vector_db = None
        
        with st.status("Initializing RAG system... This may take a moment.", expanded=True) as status:
            try:
                st.write("⚙️ Running pre-processing and setup...")
                # The setup_chatbot function is called with the default app configuration.
                chain, memory, vector_db = setup_chatbot(app_config)
                
                # Store the initialized components in the session state.
                st.session_state.chain = chain
                st.session_state.memory = memory
                st.session_state.vector_db = vector_db
                
                status.update(label="Initialization Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Initialization Failed!", state="error", expanded=True)
                st.error(f"A critical error occurred during initialization: {e}")
                # Stop the app if initialization fails, as the chatbot cannot function.
                st.stop()

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Controls & Info")
        
        # The "Clear Chat Memory" button is available once the system is running.
        if st.button("🗑️ Clear Chat Memory", use_container_width=True):
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.rerun()

        st.divider()

        # Display the current state of the conversation memory.
        with st.container(border=True):
            st.subheader("Memory State")
            if st.session_state.memory and st.session_state.memory.chat_memory.messages:
                for msg in st.session_state.memory.chat_memory.messages[-6:]:
                    msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                    st.info(f"**{msg_type}:** {msg.content[:50]}...")
            else:
                st.text("No conversation history yet.")
        
        st.divider()

        # Display key information about the system configuration.
        with st.container(border=True):
            st.subheader("System Info")
            st.markdown(f"**Model:** `{app_config.MODEL_NAME}`")
            st.markdown(f"**Embeddings:** `{app_config.EMBEDDING_MODEL}`")
            st.markdown(f"**Doc Folder:** `{app_config.RAW_DOC_FOLDER}`")

    # --- Main Chat Interface ---
    # This section now runs after the automatic initialization is successful.
    st.success("The chatbot is ready! Ask your question below.")

    # Display the existing chat messages from the session history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If the message is from the assistant and has sources, display them.
            if message["role"] == "assistant" and message.get("sources"):
                grouped_sources = defaultdict(list)
                for source in message["sources"]:
                    doc_name = os.path.basename(source.metadata.get('source', 'N/A'))
                    grouped_sources[doc_name].append(source.page_content)

                if grouped_sources:
                    st.markdown("---")
                    st.subheader("Sources:")
                    for doc_name, contents in grouped_sources.items():
                        with st.expander(f"📚 {doc_name}"):
                            for content_item in contents:
                                st.info(f"{content_item.strip()}")

    # Get the user's new input from the chat input box.
    prompt = st.chat_input("Ask your question here...")
    if prompt:
        # Add the user's message to the history and display it.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display the assistant's response.
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True):
                st.write("Retrieving relevant documents...")
                answer, sources = query_with_memory(
                    st.session_state.chain, 
                    prompt, 
                    st.session_state.vector_db
                )
                st.write("Generating final answer...")

            st.markdown(answer)

            # Display the sources used for this specific answer.
            if sources:
                grouped_sources = defaultdict(list)
                for source in sources:
                    doc_name = os.path.basename(source.metadata.get('source', 'N/A'))
                    grouped_sources[doc_name].append(source.page_content)

                if grouped_sources:
                    st.markdown("---")
                    st.subheader("Sources used for this answer:")
                    for doc_name, contents in grouped_sources.items():
                        with st.expander(f"📚 {doc_name}"):
                            for content_item in contents:
                                st.info(f"{content_item.strip()}")

        # Add the assistant's full response (answer and sources) to the message history.
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        # Rerun the script to update the chat display with the new messages.
        st.rerun()

# ==================================================================================================
# --- 3. Application Entry Point ---
# ==================================================================================================
if __name__ == "__main__":
    main()
