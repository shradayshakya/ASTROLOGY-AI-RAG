from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import os

class VectorStore:
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
        self.vectorstore = Pinecone(embedding_function=OpenAIEmbeddings(), index_name=self.index_name)

    def index_documents(self, documents):
        for doc in documents:
            self.vectorstore.add_texts([doc['text']], metadatas=[doc['metadata']])

    def query(self, query_text, top_k=5):
        return self.vectorstore.similarity_search(query_text, k=top_k)