import streamlit as st
import os
from dotenv import load_dotenv
from rag_utils import process_documents, answer_question

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG App with Llama & LangChain", layout="wide")
st.title("🔍 RAG App: Llama + LangChain + HuggingFace + Cohere Rerank")

st.markdown("""
Upload up to 2 files (PDF/TXT), or provide a URL or Wikipedia topic. Ask questions and get accurate answers using Retrieval-Augmented Generation!
""")

# File uploader (max 2 files)
uploaded_files = st.file_uploader(
    "Upload up to 2 files (PDF/TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="You can upload a maximum of 2 files."
)
if uploaded_files and len(uploaded_files) > 2:
    st.error("You can only upload up to 2 files.")
    st.stop()

# URL and Wikipedia input
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("Or enter a URL to load:")
with col2:
    wiki_topic = st.text_input("Or enter a Wikipedia topic:")

# Process documents
if st.button("Process Documents"):
    if not uploaded_files and not url and not wiki_topic:
        st.warning("Please upload at least one file or provide a URL/Wikipedia topic.")
        st.stop()
    with st.spinner("Processing documents and building vector store..."):
        docs, vectorstore = process_documents(uploaded_files, url, wiki_topic)
    st.session_state["docs"] = docs
    st.session_state["vectorstore"] = vectorstore
    st.success("Documents processed and vector store ready!")

# Question input
if "vectorstore" in st.session_state:
    question = st.text_input("Ask a question about your documents:")
    if st.button("Get Answer") and question:
        with st.spinner("Retrieving answer..."):
            answer, sources = answer_question(
                question,
                st.session_state["vectorstore"],
                st.session_state["docs"]
            )
        st.markdown(f"**Answer:** {answer}")
        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                st.markdown(f"- {src}") 