import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# Load environment variables (API Keys)
load_dotenv()

# Check for API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# Constants
DB_PATH = "./vectorstore"

def get_qa_chain():
    """
    Initializes the RAG chain:
    1. Loads the Vector DB.
    2. Sets up the LLM (Groq).
    3. Creates the Retriever (Fetch logic).
    4. Returns the QA Chain.
    """
    
    # 1. Load the Embedding Model (Must be same as Ingestion)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Load the Vector Database
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("❌ Vector DB not found. Run 'src/ingest_data.py' first.")
        
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # 3. Setup the LLM
    # Using Llama3-8b on Groq (Fast & Free tier)
    llm = ChatGroq(
        temperature=0,              # 0 means factual/deterministic, good for RAG
        model_name="llama-3.1-8b-instant",#"llama3-8b-8192",
        groq_api_key=api_key
    )

    # 4. Create the Retriever
    # search_kwargs={"k": 3} means "Fetch the top 3 most relevant text chunks"
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 5. Create the Chain
    # RetrievalQA is a classic chain that orchestrates: Question -> Retrieve -> Combine -> LLM -> Answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "Stuff" simply stuffs the docs into the prompt
        retriever=retriever,
        return_source_documents=True # Helps us see which file the answer came from
    )
    
    return qa_chain
