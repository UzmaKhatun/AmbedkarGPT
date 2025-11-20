import streamlit as st
from src.rag_engine import get_qa_chain

# --- Page Config ---
st.set_page_config(page_title="AmbedkarGPT", page_icon="📜")

# --- Header ---
st.title("📜 AmbedkarGPT")
st.markdown("Ask questions about Dr. B.R. Ambedkar's speeches/writings.")
st.divider()

# --- Initialize Chain (Cached to prevent reloading on every click) ---
@st.cache_resource
def load_chain():
    return get_qa_chain()

try:
    chain = load_chain()
except Exception as e:
    st.error(f"Failed to load system: {e}. Did you run `ingest_data.py`?")
    st.stop()

# --- Chat Interface ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is Dr. Ambedkar's view on Democracy?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                response = chain.invoke({"query": prompt})
                answer = response['result']
                
                st.markdown(answer)
                
                # Show sources in an expander
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response['source_documents']):
                        source = doc.metadata.get('source', 'Unknown')
                        content_preview = doc.page_content[:200] + "..."
                        st.write(f"**Source {i+1} ({source}):** {content_preview}")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")