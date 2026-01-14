import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL

# Load environment variables from .env file
load_dotenv()

def ingest_data(pdf_path: str = "data/BPHS.pdf"):
    """
    Loads a PDF, splits it into chunks, and ingests it into a Pinecone index.

    Args:
        pdf_path (str): The path to the PDF file to ingest.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Loading data from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        print("No documents loaded from the PDF. Check the file content.")
        return

    print(f"Loaded {len(documents)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} chunks. Initializing Pinecone...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index already exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new serverless index: {PINECONE_INDEX_NAME}")
        # Determine index dimension for common OpenAI embedding models
        # Defaults: ada-002 = 1536, text-embedding-3-small = 1536, text-embedding-3-large = 3072
        dim_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        index_dim = dim_map.get(EMBEDDING_MODEL, 1536)

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=index_dim,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    print("Initializing embeddings and vector store...")
    # Use the same embedding model for ingestion and retrieval
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Ingest data into Pinecone
    PineconeVectorStore.from_documents(
        docs, embeddings, index_name=PINECONE_INDEX_NAME
    )
    
    print("\n--- Ingestion Complete ---")
    print(f"Successfully ingested {len(docs)} chunks into the '{PINECONE_INDEX_NAME}' index.")

if __name__ == "__main__":
    ingest_data()
