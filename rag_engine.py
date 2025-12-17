import os
import shutil
import datetime
import subprocess
import glob
import json
import requests
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
PERSIST_DIR = "./storage"
DATA_DIR = "./KB"
MODELS_DIR = "./models"
CHROMA_PATH = os.path.join(PERSIST_DIR, "chroma_db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Global instances
_embedding_model = None
_chroma_client = None
_collection = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading Embedding Model...")
        # Using a solid, small model for RAG
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=MODELS_DIR)
    return _embedding_model

def get_db_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(name="pdf_knowledge_base")
    return _collection

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to KB directory with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_pdf(file_path):
    """Extracts text from PDF with page numbers."""
    reader = PdfReader(file_path)
    text_chunks = []
    file_name = os.path.basename(file_path)
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Simple chunking by page for now, can be improved to sliding window
            # Adding page metadata is crucial for citations
            chunk = {
                "id": f"{file_name}_p{i+1}",
                "text": text,
                "metadata": {
                    "source": file_name,
                    "page": i + 1
                }
            }
            text_chunks.append(chunk)
    return text_chunks

def build_index():
    """Reads all PDFs in KB, creates embeddings, and updates Vector DB."""
    collection = get_db_collection()
    model = get_embedding_model()
    
    # clear existing? For simplicity, we might just add new ones or clear all.
    # Let's clear to avoid duplicates for this simple implementation
    # Note: In production you'd track file hashes.
    try:
        # Currently no easy 'clear' in chroma api without deleting collection
        # We will delete and recreate
        global _chroma_client, _collection
        _chroma_client.delete_collection("pdf_knowledge_base")
        _collection = _chroma_client.get_or_create_collection("pdf_knowledge_base")
        collection = _collection
    except:
        pass

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    all_chunks = []
    
    for pdf_file in pdf_files:
        chunks = process_pdf(pdf_file)
        all_chunks.extend(chunks)
        
    if not all_chunks:
        return "No documents found."

    # Batch process embeddings
    texts = [chunk["text"] for chunk in all_chunks]
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    # Store explicit newlines in metadata if needed or handle in text
    
    print("Generating Embeddings...")
    embeddings = model.encode(texts).tolist()
    
    print("Storing in ChromaDB...")
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    return "Index Built Successfully"

def clear_data():
    """Clears physical files and DB."""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                os.unlink(file_path)
            except:
                pass

def query_gemini(prompt, api_key, model_name="gemini-1.5-flash"):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response.text

def query_ollama(prompt, model_name="llama3"):
    # Ensure model exists
    try:
        req = requests.post('http://localhost:11434/api/show', json={'name': model_name})
        if req.status_code != 200:
             print(f"Pulling {model_name}...")
             subprocess.run(["ollama", "pull", model_name], check=True)
    except:
        pass

    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error from Ollama: {response.text}"

def query_knowledge_base(query):
    collection = get_db_collection()
    model = get_embedding_model()
    
    # 1. Embed query
    query_embed = model.encode([query]).tolist()
    
    # 2. Retrieve top K
    results = collection.query(
        query_embeddings=query_embed,
        n_results=3
    )
    
    # 3. Construct Context
    context_text = ""
    sources = []
    
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context_text += f"\n---\nSource: {meta['source']} (Page {meta['page']})\nContent: {doc}\n"
            sources.append(meta)
            
    # 4. Generate Answer
    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
    If the answer is not in the context, say "I couldn't find the answer in the provided documents."
    
    Context:
    {context_text}
    
    Question: 
    {query}
    """
    
    provider = os.getenv("LLM_PROVIDER", "Gemini")
    
    response_text = ""
    if provider == "Gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        try:
            response_text = query_gemini(prompt, api_key, model_name)
        except Exception as e:
            response_text = f"Gemini Error: {e}"
    else:
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        try:
            response_text = query_ollama(prompt, model_name)
        except Exception as e:
            response_text = f"Ollama Error: {e}"
            
    return {
        "response": response_text,
        "source_nodes": sources
    }

def generate_quiz(topic, num_questions=5, q_type="MCQ"):
    # Reuse context retrieval logic? Or generic?
    # Usually quiz is broader. Let's do a broad retrieval or just ask LLM to use its knowledge 
    # BUT user wants "based on documents". 
    # We'll do a vector search for the topic to get relevant context.
    
    collection = get_db_collection()
    model = get_embedding_model()
    
    query_embed = model.encode([topic]).tolist()
    results = collection.query(
        query_embeddings=query_embed,
        n_results=5 # More context for quiz
    )
    
    context_text = ""
    if results['documents']:
        for doc in results['documents'][0]:
            context_text += f"{doc}\n"
            
    prompt = ""
    if q_type == "MCQ":
        prompt = (
            f"Based on the text below, generate {num_questions} Multiple Choice Questions (MCQ) about '{topic}'.\n"
            f"Format:\n1. Question\na) Option\nb) Option\nc) Option\nd) Option\nCorrect Answer: X\nExplanation: ...\n\n"
            f"Text:\n{context_text}"
        )
    else:
        prompt = (
            f"Based on the text below, generate {num_questions} descriptive questions and answers about '{topic}'.\n"
            f"Text:\n{context_text}"
        )

    provider = os.getenv("LLM_PROVIDER", "Gemini")
    if provider == "Gemini":
        return query_gemini(prompt, os.getenv("GOOGLE_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    else:
        return query_ollama(prompt, os.getenv("OLLAMA_MODEL", "llama3"))
