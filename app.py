import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azure_cognitive_search import AzureCognitiveSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import uuid
from typing import List
import tempfile
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Document Chat",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def initialize_vector_store(openai_api_key: str):
    """Initialize Azure Cognitive Search vector store"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = AzureCognitiveSearch(
            azure_search_endpoint=os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME"),
            embedding_function=embeddings.embed_query,
        )
        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

def process_document(file, vector_store):
    """Process uploaded document and add to vector store"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name

        # Extract text (you may want to enhance this based on your needs)
        text = ""
        with open(file_path, 'rb') as pdf_file:
            # Add your PDF text extraction logic here
            text = pdf_file.read().decode('utf-8', errors='ignore')

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Add to vector store
        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"source": file.name, "chunk_id": str(i)} for i in range(len(chunks))]
        )

        # Cleanup
        os.unlink(file_path)
        return True
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False

def initialize_conversation(vector_store, openai_api_key: str):
    """Initialize the conversation chain"""
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4",
        openai_api_key=openai_api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),
        memory=memory,
        verbose=True
    )
    
    return conversation

# Sidebar for API keys and configuration
with st.sidebar:
    st.title("Configuration")
    
    # OpenAI API Key input
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key here. It will not be stored.",
        key="openai_api_key"
    )
    
    # Document upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

# Main content
st.title("💬 Chat with Your Documents")

# Initialize vector store if API key is provided
if openai_api_key:
    if st.session_state.vector_store is None:
        with st.spinner("Initializing vector store..."):
            st.session_state.vector_store = initialize_vector_store(openai_api_key)

    # Process uploaded documents
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                if process_document(file, st.session_state.vector_store):
                    st.success(f"Successfully processed {file.name}")

    # Initialize conversation if needed
    if st.session_state.conversation is None and st.session_state.vector_store is not None:
        st.session_state.conversation = initialize_conversation(
            st.session_state.vector_store,
            openai_api_key
        )

    # Chat interface
    if st.session_state.conversation is not None:
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({"question": prompt})
                    st.write(response["answer"])
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response["answer"]}
                    )

else:
    st.warning("Please enter your OpenAI API key in the sidebar to start.")
