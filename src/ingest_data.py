import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_PATH = "./data"
DB_PATH = "./vectorstore"

def create_vector_db():
    """
    Reads text files, creates embeddings, and stores them in ChromaDB.
    """
    print(f"🔄 Checking for data in {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Data directory not found. Run data_setup.py first.")
        return

    # 1. Load Data
    # We use DirectoryLoader to load all .txt files in the data folder
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")

    # 2. Split Text (Chunking)
    # We split text so the AI can process small pieces at a time
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Characters per chunk
        chunk_overlap=50     # Overlap ensures context isn't lost at cut points
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")

    # 3. Clear old database (optional, but good for clean testing)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # 4. Create Embeddings & Store in ChromaDB
    # Using the model specified in the assignment requirements
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("🔄 Creating Vector Store (this may take a moment)...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"🎉 Success! Vector Database created at {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()

