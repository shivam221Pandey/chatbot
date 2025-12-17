import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from rag_engine import (
    build_index, 
    save_uploaded_file, 
    query_knowledge_base, 
    generate_quiz, 
    clear_data
)

# Page Config
st.set_page_config(
    page_title="Local AI Knowledge Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatInput {
        border-radius: 20px;
    }
    div[data-testid="stCard"] {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Local AI Knowledge Agent")
st.markdown("### Your personal Research Assistant & Examiner")

# Sidebar for Configuration & Data
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load Config from Env
    provider = os.getenv("LLM_PROVIDER", "Gemini")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if provider == "Gemini":
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
        if not api_key:
             st.error("Missing GOOGLE_API_KEY in .env")
    else:
        model_name = os.getenv("OLLAMA_MODEL", "llama3")

    st.info(f"Using **{provider}** ({model_name})")
    
    st.divider()
    
    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Ingest & Train", type="primary"):
        if uploaded_files:
            with st.spinner("Processing Documents..."):
                # Initialize Settings first to ensure embed model is ready
                try:
                    for f in uploaded_files:
                        save_uploaded_file(f)
                    
                    # Rebuild Index
                    st.session_state.index = build_index() # Returns status string
                    st.session_state.index_built = True
                    st.success("Knowledge Base Updated Successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload files first.")

    if st.button("Reset Knowledge Base"):
        clear_data()
        st.session_state.index = None
        st.success("Reset Complete.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to answer questions based on your PDFs."}]

# We use a simple flag for index existence in this new version
if "index_built" not in st.session_state:
    st.session_state.index_built = False # Will rely on file check or explicit build

# Interface Tabs
tab1, tab2 = st.tabs(["💬 Chat & Research", "📝 Quiz Generator"])

# --- TAB 1: Chat Interface ---
with tab1:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if not st.session_state.get("index_built"):
                # Ideally check if DB exists using rag_engine check, but let's assume if user says ingest it works
                # Or we can leniently allow query and let it return "No documents"
                pass 

            with st.spinner("Thinking..."):
                try:
                    response = query_knowledge_base(prompt) # No index arg needed
                    answer_text = response["response"]
                    st.markdown(answer_text)
                    
                    # Show Citations / Sources
                    if response.get("source_nodes"):
                        with st.expander("📚 Sources & Page Numbers"):
                            for node in response["source_nodes"]:
                                page_label = node.get('page', 'N/A')
                                file_name = node.get('source', 'Unknown')
                                
                                st.markdown(f"**{file_name}** (Page {page_label})")
                                st.divider()
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                except Exception as e:
                    err_msg = f"An error occurred: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

# --- TAB 2: Quiz Generator ---
with tab2:
    st.header("Generate Practice Questions")
    
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Enter Topic for Questions", placeholder="e.g., Photosynthesis, Chapter 1...")
    with col2:
        q_type = st.selectbox("Question Type", ["MCQ", "Descriptive"])
        num_q = st.slider("Number of Questions", 1, 10, 5)
        
    if st.button("Generate Questions"):
        if not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Generating {q_type}s for '{topic}'..."):
                try:
                    result_text = generate_quiz(topic, num_q, q_type) # Removed index arg
                    st.markdown("### Generated Questions")
                    st.markdown(result_text)
                except Exception as e:
                    st.error(f"Error: {e}")
