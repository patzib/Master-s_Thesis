# app.py

# ==================================================================================================
# --- 1. Import Dependencies ---
# ==================================================================================================
# --- Standard Library Imports ---
import os
from collections import defaultdict

# --- Third-Party Library Imports ---
# Streamlit is used for creating the web-based user interface for the chatbot.
import streamlit as st
# LangChain core message types are used for managing conversation history.
from langchain_core.messages import HumanMessage

# --- Local Application Imports ---
# `setup_chatbot` is the main function from our RAG pipeline script that initializes all components.
from rag_setup import setup_chatbot
# `config` provides centralized access to model names and file paths.
from config import MODEL_NAME, EMBEDDING_MODEL, RAW_DOC_FOLDER
# `rag_utils` contains helper functions, such as the one for dynamic query analysis.
from rag_utils import get_lecture_filter_from_query

# ==================================================================================================
# --- 2. Core Application Logic ---
# ==================================================================================================

def query_with_memory(chain, question: str, vector_db):
    """
    Executes a query against the RAG chain with dynamic metadata filtering.

    This function represents a key innovation in the RAG pipeline. Instead of using a static
    retriever, it dynamically modifies the retriever's search parameters based on the content
    of the user's query. This allows for more targeted document retrieval. For instance, if a
    user's question mentions a specific lecture, the retriever is temporarily configured to
    search only within documents associated with that lecture's metadata.

    This dynamic filtering is applied to the `base_retriever` of the `ContextualCompressionRetriever`,
    ensuring that the subsequent re-ranking step still functions as intended.

    Args:
        chain: The initialized ConversationalRetrievalChain instance.
        question: The user's input question.
        vector_db: The Chroma vector database instance.

    Returns:
        A tuple containing the generated answer (str) and a list of the source documents used.
    """
    try:
        # Step 1: Analyze the user's query to detect any mention of a specific lecture.
        # This uses an LLM to extract a lecture number, if present.
        lecture_filter = get_lecture_filter_from_query(question)

        # Step 2: Define the base search parameters for the retriever.
        # This will fetch 6 documents for the re-ranker to process.
        search_kwargs = {"k": 6}

        # Step 3: If a lecture number was identified, add a metadata filter to the search parameters.
        # This instructs the vector store to perform a filtered similarity search.
        if lecture_filter > 0:
            st.write(f"Applying filter for Lecture {lecture_filter}...")
            search_kwargs["filter"] = {"lecture_number": lecture_filter}

        # Step 4: Dynamically update the base retriever within the chain's compression retriever.
        # The chain's `retriever` is the `ContextualCompressionRetriever`. We modify its
        # `base_retriever` attribute on-the-fly. This is a powerful technique that preserves
        # the re-ranking logic while changing the initial retrieval pool.
        chain.retriever.base_retriever = vector_db.as_retriever(
            search_kwargs=search_kwargs
        )

        # Step 5: Invoke the chain with the user's question. The chain will now use the
        # dynamically configured retriever for this specific query.
        result = chain.invoke({"question": question})
        answer = result.get("answer", "Sorry, I couldn't find an answer.")

        # The re-ranker will have reduced the initial 6 documents to the top 3.
        sources = result.get("source_documents", [])

        return answer, sources

    except Exception as e:
        # Log any errors that occur during the query process to the Streamlit UI.
        st.error(f"Error during query execution: {e}")
        return "An error occurred while processing your question.", []

def main():
    """
    The main function that defines and runs the Streamlit application user interface.

    This function orchestrates the entire front-end of the application. It handles:
    - Page configuration and layout.
    - Initialization of the RAG system using Streamlit's session state to ensure it runs only once.
    - Definition of the UI components, including the main chat interface and a sidebar for controls.
    - Management of the chat history and display of messages.
    - Handling of user input and the subsequent call to the query function.
    """
    st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

    st.title("MLME Chatbot")
    st.markdown("Ask questions about the course 'Machine Learning for Management Decisions'!")

    # --- Session State Management ---
    # Streamlit's session state is used to persist variables across user interactions (reruns).
    # This is crucial for maintaining the chat history and the initialized RAG chain
    # without having to reload the entire system on every action.
    if 'chain' not in st.session_state:
        st.session_state.chain = None
        st.session_state.memory = None
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.initialized = False

    # --- System Initialization ---
    # This block runs only once when the application starts. The `initialized` flag
    # in the session state prevents the costly setup process from re-running.
    if not st.session_state.initialized:
        # `st.status` provides visual feedback to the user during the setup process.
        with st.status("Initializing RAG system, please wait...", expanded=True) as status:
            st.write("⚙️ Running pre-processing and setup...")
            try:
                # Call the main setup function from `rag_setup.py`.
                chain, memory, vector_db = setup_chatbot()
                # Store the initialized components in the session state for later use.
                st.session_state.chain = chain
                st.session_state.memory = memory
                st.session_state.vector_db = vector_db
                st.session_state.initialized = True # Mark initialization as complete.
                status.update(label="Initialization Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Initialization Failed!", state="error", expanded=True)
                st.error(f"A critical error occurred during setup: {e}")
                # If initialization fails, the application cannot proceed.
                st.stop()

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Controls & Info")

        if st.session_state.initialized:
            if st.button("🗑️ Clear Chat Memory", use_container_width=True):
                # This button allows the user to reset the conversation.
                if st.session_state.memory:
                    st.session_state.memory.clear()
                st.session_state.messages = []
                st.success("Chat memory cleared!")
                st.rerun() # Rerun the app to reflect the cleared state.

        st.divider()

        # Display the current state of the conversation memory for debugging/demonstration.
        with st.container(border=True):
            st.subheader("Memory State")
            if st.session_state.memory and st.session_state.memory.chat_memory.messages:
                # Show a snippet of the last few messages.
                for msg in st.session_state.memory.chat_memory.messages[-6:]:
                    msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                    st.info(f"**{msg_type}:** {msg.content[:50]}...")
            else:
                st.text("No conversation history yet.")

        st.divider()

        # Display key system configuration details.
        with st.container(border=True):
            st.subheader("System Info")
            st.markdown(f"**Model:** `{MODEL_NAME}`")
            st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
            st.markdown(f"**Doc Folder:** `{RAW_DOC_FOLDER}`")

    # --- Main Chat Interface ---
    # This part of the UI is only rendered if the system has been successfully initialized.
    if st.session_state.initialized:
        st.success("The chatbot is ready! Ask your question below.")

        # Display the entire chat history from the session state.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # If a message is from the assistant and has sources, display them in an expander.
                if message["role"] == "assistant" and message.get("sources"):
                    # Group sources by the original document name for a cleaner presentation.
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

        # Handle new user input via the chat input widget.
        if prompt := st.chat_input("Ask your question here..."):
            # Add user's message to history and display it.
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process the user's query and display the assistant's response.
            with st.chat_message("assistant"):
                with st.status("Thinking...", expanded=True):
                    st.write("Retrieving and re-ranking documents...")
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