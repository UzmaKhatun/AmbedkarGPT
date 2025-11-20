import sys
import time
from src.rag_engine import get_qa_chain

def main():
    print("----------------------------------------------------------")
    print("🤖 AmbedkarGPT - Command Line Interface")
    print("----------------------------------------------------------")
    print("Initializing system...")
    
    try:
        qa_chain = get_qa_chain()
        print("✅ System Ready! Type 'exit' or 'quit' to stop.")
        print("----------------------------------------------------------")
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return

    while True:
        query = input("\n❓ Enter your question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue

        print("🤔 Thinking...")
        start_time = time.time()
        
        try:
            # Get response from RAG chain
            response = qa_chain.invoke({"query": query})
            end_time = time.time()
            
            result = response['result']
            sources = response['source_documents']
            
            # Print Result
            print(f"\n💡 Answer ({round(end_time - start_time, 2)}s):")
            print(result)
            
            # Print Sources (Optional but helpful for debugging)
            print("\n📄 Sources Used:")
            for doc in sources:
                source_name = doc.metadata.get('source', 'Unknown')
                print(f" - {source_name}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()