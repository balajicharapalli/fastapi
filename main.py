import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 🔹 Load environment variables
load_dotenv()

# 🔹 Initialize FastAPI
app = FastAPI(title="MITS Alumni RAG API")

# 🔹 Load alumni data
loader = TextLoader("alumni.txt")
docs = loader.load()

# 🔹 Split docs
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 🔹 Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

# 🔹 LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 🔹 RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True
)

# 🔹 Request schema
class QueryRequest(BaseModel):
    question: str

# 🔹 API Endpoint
@app.post("/ask")
def ask_question(req: QueryRequest):
    result = qa.invoke(req.question)
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }

