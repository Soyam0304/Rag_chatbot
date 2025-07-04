import streamlit as st
import os
from dotenv import load_dotenv
from rag_utils import process_documents, answer_question

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG App with Llama & LangChain", layout="wide")

# Sidebar image and info
with st.sidebar:
    st.image(
        "https://i.pinimg.com/736x/e0/aa/ce/e0aace4e8ac951195fbbd1a97b0c1d87.jpg",
        width=180,
        caption="AI Document Q&A Assistant"
    )
    st.markdown(
        """
        **About this Chatbot**
        
        - 🤖 Ask questions about your uploaded documents, URLs, or Wikipedia topics.
        - 📄 Supports PDF, TXT, web, and Wikipedia sources.
        - 🧠 Uses advanced AI (RAG, Llama, Cohere, HuggingFace) for accurate answers.
        - 💬 Chat history is maintained for each document session.
        """
    )

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

# Detect if new documents are uploaded or source changed
new_docs_uploaded = False
if (
    uploaded_files and (
        "last_uploaded_filenames" not in st.session_state or
        [f.name for f in uploaded_files] != st.session_state.get("last_uploaded_filenames", [])
    )
) or (
    url and url != st.session_state.get("last_url", "")
) or (
    wiki_topic and wiki_topic != st.session_state.get("last_wiki_topic", "")
):
    new_docs_uploaded = True

# Reset memory if new docs/source
if new_docs_uploaded:
    st.session_state["chat_history"] = []
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("docs", None)
    st.session_state["last_uploaded_filenames"] = [f.name for f in uploaded_files] if uploaded_files else []
    st.session_state["last_url"] = url
    st.session_state["last_wiki_topic"] = wiki_topic

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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat interface
if "vectorstore" in st.session_state:
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        st.session_state["chat_history"].append(("User", user_input))
        with st.spinner("Retrieving answer..."):
            answer, sources = answer_question(
                user_input,
                st.session_state["vectorstore"],
                st.session_state["docs"]
            )
        response = f"**Answer:** {answer}"
        if sources:
            response += "\n\n**Sources:**\n" + "\n".join(f"- {src}" for src in sources)
        st.session_state["chat_history"].append(("Agent", response))

    # Show chat
    for sender, message in st.session_state["chat_history"]:
        with st.chat_message(sender):
            st.markdown(message, unsafe_allow_html=True) 