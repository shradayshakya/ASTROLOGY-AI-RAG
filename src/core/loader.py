from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import os

class DocumentLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_documents(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        return documents

class VectorStoreManager:
    def __init__(self, pinecone_api_key, index_name):
        pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
        self.index = Pinecone(index_name=index_name, embedding_function=OpenAIEmbeddings())

    def index_documents(self, documents):
        for doc in documents:
            self.index.add_texts([doc.page_content], metadatas=[{"source": doc.metadata['source']}])

    def query(self, query_text, top_k=5):
        return self.index.query(query_text, top_k=top_k)

def load_and_index_documents(pdf_path, pinecone_api_key, index_name):
    loader = DocumentLoader(pdf_path)
    documents = loader.load_documents()
    
    vector_store_manager = VectorStoreManager(pinecone_api_key, index_name)
    vector_store_manager.index_documents(documents)