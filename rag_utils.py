import os
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredURLLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import cohere
from langchain.schema import HumanMessage

load_dotenv()

# API keys from .env
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cohere client
co = cohere.Client(cohere_api_key)

def process_documents(uploaded_files, url, wiki_topic):
    docs = []
    # Load PDF and TXT files
    for file in uploaded_files or []:
        if file.name.endswith(".pdf"):
            suffix = ".pdf"
        elif file.name.endswith(".txt"):
            suffix = ".txt"
        else:
            continue  # skip unsupported files
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path)
        docs.extend(loader.load())
    # Load from URL
    if url:
        loader = UnstructuredURLLoader(urls=[url])
        docs.extend(loader.load())
    # Load from Wikipedia
    if wiki_topic:
        loader = WikipediaLoader(query=wiki_topic, lang="en")
        docs.extend(loader.load())
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # Build vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return docs, vectorstore

def answer_question(question, vectorstore, docs, k=6):
    # Retrieve top-k docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(question)
    # Rerank with Cohere
    texts = [doc.page_content for doc in retrieved_docs]
    rerank_results = co.rerank(query=question, documents=texts, top_n=3)
    reranked = [retrieved_docs[r.index] for r in rerank_results.results]
    # Prepare context
    context = "\n\n".join([doc.page_content for doc in reranked])
    # Llama (ChatGroq) LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.2
    )
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    response = llm([HumanMessage(content=prompt)])
    answer = response.content
    # Collect sources
    sources = [getattr(doc, 'metadata', {}).get('source', 'N/A') for doc in reranked]
    return answer, sources 