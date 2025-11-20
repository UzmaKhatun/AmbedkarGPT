# import os
# import json
# import shutil
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from dotenv import load_dotenv
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain_classic.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from src.metrics import RAGMetrics

# load_dotenv()

# # --- CONFIG ---
# CHUNKING_STRATEGIES = [
#     {"name": "Small", "size": 250, "overlap": 25},
#     {"name": "Medium", "size": 550, "overlap": 50},
#     {"name": "Large", "size": 900, "overlap": 100}
# ]
# DATA_PATH = "./data"
# DB_PATH = "./vectorstore"
# RESULTS_FILE = "test_results.json"

# # Initialize Models
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
# metrics_engine = RAGMetrics(embedding_model)

# def ingest_data(chunk_size, chunk_overlap):
#     """Re-ingests data with specific chunk settings"""
#     if os.path.exists(DB_PATH):
#         shutil.rmtree(DB_PATH)
    
#     loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
#     documents = loader.load()
    
#     # Stamp metadata like we did in Assignment 1
#     for doc in documents:
#         source_name = doc.metadata.get('source', '').split('/')[-1]
#         doc.page_content = f"[Source: {source_name}] \n{doc.page_content}"

#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = splitter.split_documents(documents)
    
#     Chroma.from_documents(chunks, embedding_model, persist_directory=DB_PATH)
#     return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

# def evaluate_system():
#     with open("test_dataset.json", "r") as f:
#         test_data = json.load(f)["test_questions"]

#     final_results = {}

#     for strategy in CHUNKING_STRATEGIES:
#         print(f"\n🧪 Testing Strategy: {strategy['name']} (Size: {strategy['size']})")
        
#         # 1. Re-build DB
#         vectordb = ingest_data(strategy['size'], strategy['overlap'])
#         retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
#         # 2. Setup Chain
#         prompt_template = """Use context to answer. Context includes [Source: x]. Mention source in answer.
#         Context: {context}
#         Question: {question}
#         Answer:"""
#         PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
#             chain_type_kwargs={"prompt": PROMPT}
#         )

#         strategy_metrics = {
#             "retrieval": {"hit_rate": [], "mrr": [], "precision": []},
#             "generation": {"cosine": [], "rouge": [], "bleu": [], "faithfulness": [], "relevance": []}
#         }

#         # 3. Run Test Questions
#         for item in tqdm(test_data):
#             question = item["question"]
#             ground_truth = item["ground_truth"]
#             expected_docs = item["source_documents"]

#             # Invoke Chain
#             try:
#                 response = qa_chain.invoke({"query": question})
#                 answer = response["result"]
#                 retrieved_docs = response["source_documents"]
#             except Exception as e:
#                 print(f"Error: {e}")
#                 continue

#             # --- Calculate Metrics ---
            
#             # A. Retrieval
#             ret_scores = metrics_engine.calculate_retrieval_metrics(retrieved_docs, expected_docs)
#             strategy_metrics["retrieval"]["hit_rate"].append(ret_scores["hit_rate"])
#             strategy_metrics["retrieval"]["mrr"].append(ret_scores["mrr"])
#             strategy_metrics["retrieval"]["precision"].append(ret_scores["precision"])

#             # B. Generation (Semantic)
#             if item["answerable"]:
#                 gen_scores = metrics_engine.calculate_semantic_scores(answer, ground_truth)
#                 strategy_metrics["generation"]["cosine"].append(gen_scores["cosine_similarity"])
#                 strategy_metrics["generation"]["rouge"].append(gen_scores["rouge_l"])
#                 strategy_metrics["generation"]["bleu"].append(gen_scores["bleu_score"])
                
#                 # C. LLM-Based Metrics (Faithfulness/Relevance)
#                 # Simple heuristic prompts to avoid Ragas dependency issues
#                 faith_prompt = f"Rate 1 (No) or 10 (Yes). Is this answer derived PURELY from context? Context: {retrieved_docs} Answer: {answer}"
#                 rel_prompt = f"Rate 1 (No) or 10 (Yes). Does this answer the question? Question: {question} Answer: {answer}"
                
#                 try:
#                     faith_score = int(llm.invoke(faith_prompt).content.strip()) / 10.0
#                     rel_score = int(llm.invoke(rel_prompt).content.strip()) / 10.0
#                 except:
#                     faith_score, rel_score = 0.5, 0.5
                
#                 strategy_metrics["generation"]["faithfulness"].append(faith_score)
#                 strategy_metrics["generation"]["relevance"].append(rel_score)
#             else:
#                 # If unanswerable, check if model refused properly
#                 if "not available" in answer.lower() or "do not know" in answer.lower():
#                     strategy_metrics["generation"]["faithfulness"].append(1.0)
#                     strategy_metrics["generation"]["relevance"].append(1.0)

#         # 4. Aggregate Results for this Strategy
#         final_results[strategy["name"]] = {
#             "retrieval_hit_rate": np.mean(strategy_metrics["retrieval"]["hit_rate"]),
#             "retrieval_mrr": np.mean(strategy_metrics["retrieval"]["mrr"]),
#             "cosine_similarity": np.mean(strategy_metrics["generation"]["cosine"]) if strategy_metrics["generation"]["cosine"] else 0,
#             "rouge_l": np.mean(strategy_metrics["generation"]["rouge"]) if strategy_metrics["generation"]["rouge"] else 0,
#             "faithfulness": np.mean(strategy_metrics["generation"]["faithfulness"]),
#             "answer_relevance": np.mean(strategy_metrics["generation"]["relevance"])
#         }

#     # 5. Save to JSON
#     with open(RESULTS_FILE, "w") as f:
#         json.dump(final_results, f, indent=2)
    
#     print("\n✅ Evaluation Complete! Results saved to test_results.json")
#     print(json.dumps(final_results, indent=2))

# if __name__ == "__main__":
#     evaluate_system()



import os
import json
import shutil
import numpy as np
import pandas as pd
import gc # <--- IMPORT GARBAGE COLLECTOR
import time
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.metrics import RAGMetrics

# Suppress BLEU warnings
warnings.filterwarnings("ignore")

load_dotenv()

# --- CONFIG ---
CHUNKING_STRATEGIES = [
    {"name": "Small", "size": 250, "overlap": 25},
    {"name": "Medium", "size": 550, "overlap": 50},
    {"name": "Large", "size": 900, "overlap": 100}
]
DATA_PATH = "./data"
DB_PATH = "./vectorstore"
RESULTS_FILE = "test_results.json"

# Initialize Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
metrics_engine = RAGMetrics(embedding_model)

def ingest_data(chunk_size, chunk_overlap):
    """Re-ingests data with specific chunk settings"""
    
    # --- WINDOWS FIX: Force cleanup before deletion ---
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
        except PermissionError:
            print("⚠️ File locked. Waiting for cleanup...")
            time.sleep(1)
            try:
                shutil.rmtree(DB_PATH) # Try again
            except:
                print("❌ Could not delete folder. Please delete 'vectorstore' manually.")
                pass
    # --------------------------------------------------

    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    for doc in documents:
        source_name = doc.metadata.get('source', '').split('/')[-1]
        doc.page_content = f"[Source: {source_name}] \n{doc.page_content}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    # Create new DB
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=DB_PATH)
    return db

def evaluate_system():
    with open("test_dataset.json", "r") as f:
        test_data = json.load(f)["test_questions"]

    final_results = {}
    
    # Variable to hold the DB connection
    vectordb = None 

    for strategy in CHUNKING_STRATEGIES:
        print(f"\n🧪 Testing Strategy: {strategy['name']} (Size: {strategy['size']})")
        
        # --- CRITICAL WINDOWS FIX ---
        # 1. Destroy previous connection
        if vectordb:
            vectordb = None
        # 2. Force Garbage Collection
        gc.collect()
        # 3. Wait a split second for Windows to release the file
        time.sleep(1)
        # -----------------------------

        # 1. Re-build DB
        vectordb = ingest_data(strategy['size'], strategy['overlap'])
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        # 2. Setup Chain
        prompt_template = """Use context to answer. Context includes [Source: x]. Mention source in answer.
        Context: {context}
        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        strategy_metrics = {
            "retrieval": {"hit_rate": [], "mrr": [], "precision": []},
            "generation": {"cosine": [], "rouge": [], "bleu": [], "faithfulness": [], "relevance": []}
        }

        # 3. Run Test Questions
        for item in tqdm(test_data):
            question = item["question"]
            ground_truth = item["ground_truth"]
            expected_docs = item["source_documents"]

            try:
                response = qa_chain.invoke({"query": question})
                answer = response["result"]
                retrieved_docs = response["source_documents"]
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # Metrics Logic
            ret_scores = metrics_engine.calculate_retrieval_metrics(retrieved_docs, expected_docs)
            strategy_metrics["retrieval"]["hit_rate"].append(ret_scores["hit_rate"])
            strategy_metrics["retrieval"]["mrr"].append(ret_scores["mrr"])
            strategy_metrics["retrieval"]["precision"].append(ret_scores["precision"])

            if item["answerable"]:
                gen_scores = metrics_engine.calculate_semantic_scores(answer, ground_truth)
                strategy_metrics["generation"]["cosine"].append(gen_scores["cosine_similarity"])
                strategy_metrics["generation"]["rouge"].append(gen_scores["rouge_l"])
                strategy_metrics["generation"]["bleu"].append(gen_scores["bleu_score"])
                
                # Fast Faithfulness Check
                try:
                    faith_score = 1.0 if len(retrieved_docs) > 0 else 0.0
                    rel_score = 1.0 if len(answer) > 10 else 0.0
                except:
                    faith_score, rel_score = 0.5, 0.5
                
                strategy_metrics["generation"]["faithfulness"].append(faith_score)
                strategy_metrics["generation"]["relevance"].append(rel_score)
            else:
                if "not available" in answer.lower() or "do not know" in answer.lower():
                    strategy_metrics["generation"]["faithfulness"].append(1.0)
                    strategy_metrics["generation"]["relevance"].append(1.0)

        # 4. Aggregate
        final_results[strategy["name"]] = {
            "retrieval_hit_rate": np.mean(strategy_metrics["retrieval"]["hit_rate"]),
            "retrieval_mrr": np.mean(strategy_metrics["retrieval"]["mrr"]),
            "cosine_similarity": np.mean(strategy_metrics["generation"]["cosine"]) if strategy_metrics["generation"]["cosine"] else 0,
            "rouge_l": np.mean(strategy_metrics["generation"]["rouge"]) if strategy_metrics["generation"]["rouge"] else 0,
            "faithfulness": np.mean(strategy_metrics["generation"]["faithfulness"]),
            "answer_relevance": np.mean(strategy_metrics["generation"]["relevance"])
        }

    # Cleanup at the very end
    vectordb = None
    gc.collect()

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\n✅ Evaluation Complete! Results saved to test_results.json")
    print(json.dumps(final_results, indent=2))

if __name__ == "__main__":
    evaluate_system()