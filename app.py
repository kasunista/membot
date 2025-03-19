import os
import streamlit as st
import base64
from openai import AzureOpenAI

# Page configuration
st.set_page_config(page_title="Meeting Minutes AI Assistant", page_icon="📝", layout="wide")

# App title
st.title("Meeting Minutes AI Assistant")
st.markdown("Ask questions about your meeting minutes documents.")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for Azure OpenAI configuration
with st.sidebar:
    st.header("Configuration")
    
    # Azure OpenAI Configuration
    azure_endpoint = st.text_input("Azure OpenAI Endpoint", value=os.getenv("ENDPOINT_URL", "https://momgptista.openai.azure.com/"), key="azure_endpoint")
    azure_deployment = st.text_input("Azure OpenAI Deployment Name", value=os.getenv("DEPLOYMENT_NAME", "gpt-4o"), key="azure_deployment")
    azure_api_key = st.text_input("Azure OpenAI API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password", key="azure_api_key")
    
    # Azure AI Search Configuration
    search_endpoint = st.text_input("Azure AI Search Endpoint", value=os.getenv("SEARCH_ENDPOINT", "https://momaisearchista.search.windows.net"), key="search_endpoint")
    search_key = st.text_input("Azure AI Search Key", value=os.getenv("SEARCH_KEY", ""), type="password", key="search_key")
    search_index = st.text_input("Azure AI Search Index Name", value=os.getenv("SEARCH_INDEX_NAME", "firstindex"), key="search_index")
    
    # Search Controls
    st.subheader("Search Controls")
    
    # Strictness slider (1-5)
    strictness = st.slider(
        "Strictness (1-5)", 
        min_value=1, 
        max_value=5, 
        value=3, 
        help="Controls how strictly the model adheres to the retrieved content. Higher values = stricter adherence to data."
    )
    
    # Retrieved documents slider (3-20)
    top_n_documents = st.slider(
        "Retrieved documents (3-20)", 
        min_value=3, 
        max_value=20, 
        value=5, 
        help="Number of documents to retrieve from the search index."
    )
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="""Follow the instructions below to respond exclusively using information found in the meeting minutes PDF files provided by the user. Only reference these documents explicitly. If the user requires information outside of these PDFs, you may politely indicate that the answer cannot be found within the provided documents.

# Instructions for Task

Answer the user's queries by extracting information from the meeting minutes PDFs that have been provided. Do not introduce external information or assumptions. If information requested by the user is not available within the provided meeting minutes, explicitly state that the information cannot be found within the provided files.

# Steps

1. Process the content of the provided meeting minutes PDFs.
   - Extract and interpret relevant sections based on the user's query.
2. Reference specific details from the meeting minutes to support your response.
   - Where applicable, cite page numbers, sections, or headings from the PDF to ensure clarity and traceability.
3. If the user's question cannot be answered using the PDFs:
   - Clearly state: "This information is not available in the meeting minutes provided."
4. Focus on clarity and conciseness in your responses.

# Output Format

1. Begin the response with a direct and clear answer to the user's query.
2. For supported answers, provide:
   - Key information from the meeting minutes.
   - Any relevant page numbers, headings, or contextual references from the PDFs.
3. For unsupported answers:
   - State, "This information is not available in the meeting minutes provided."
4. Do not output content in code blocks unless explicitly requested.
""",
        height=300,
    )
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User input
user_input = st.chat_input("Ask a question about your meeting minutes...")

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Display assistant thinking
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Thinking...")
        
        try:
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version="2024-05-01-preview"
            )
            
            # Prepare the chat messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Add chat history to messages (limit to last 10 messages)
            for msg in st.session_state.chat_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
            # Generate completion
            completion = client.chat.completions.create(
                model=azure_deployment,
                messages=messages,
                max_tokens=4096,
                temperature=0.3,
                top_p=0.98,
                frequency_penalty=0,
                presence_penalty=0.2,
                stop=None,
                stream=False,
                extra_body={
                    "data_sources": [{
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": search_endpoint,
                            "index_name": search_index,
                            "semantic_configuration": "default",
                            "query_type": "simple",
                            "fields_mapping": {},
                            "in_scope": True,
                            "role_information": system_prompt,
                            "filter": None,
                            "strictness": strictness,  # Using the slider value
                            "top_n_documents": top_n_documents,  # Using the slider value
                            "authentication": {
                                "type": "api_key",
                                "key": search_key
                            }
                        }
                    }]
                }
            )
            
            # Get the response
            response = completion.choices[0].message.content
            
            # Update the placeholder with the response
            message_placeholder.write(response)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

# Add a footer
st.markdown("---")
st.markdown("Powered by Azure OpenAI and Azure AI Search")
