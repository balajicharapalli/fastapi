import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()   # this will read your .env file into os.environ


# 🔹 Load Alumni Data
loader = TextLoader("alumni.txt")
docs = loader.load()

# 🔹 Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 🔹 Create embeddings using Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Create FAISS vectorstore (in memory)
db = FAISS.from_documents(chunks, embeddings)

# 🔹 Initialize Groq LLM (needs GROQ API key in env var)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("⚠️ Please set your GROQ_API_KEY as an environment variable.")
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 🔹 Create RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True
)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="MITS Alumni RAG Bot", layout="centered")
st.title("🎓 MITS Alumni RAG Assistant")

user_query = st.text_input("Ask a question about MITS Alumni, Placements, Events, or Scholarships:")

if user_query:
    with st.spinner("Thinking..."):
        result = qa.invoke(user_query)
        st.subheader("🤖 Answer:")
        st.write(result["result"])

        with st.expander("📚 Sources"):
            for doc in result["source_documents"]:
                st.write(doc.page_content[:300] + "...")
