from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY

import tempfile
import os


from langchain_classic.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())

class SimpleRAG:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=GROQ_API_KEY
        )

        self.vectorstore = None
        self.qa_chain = None

    def ingest_document(self, file_bytes: bytes):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            chunks = self.text_splitter.split_documents(docs)

            self.vectorstore = FAISS.from_documents(
                chunks,
                self.embeddings
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever()
            )

        finally:
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def ask(self, question: str):

        if not self.qa_chain:
            return "Upload document first."

        response = self.qa_chain.invoke(question)
        return response
