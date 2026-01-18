import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from src.embedding_factory import get_embedding_model
from pinecone import Pinecone, ServerlessSpec

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME 

# Load environment variables from .env file
load_dotenv()


def ingest_data(pdf_path: str = "data/Brihat_Parashara_Hora_Shastra.pdf"):
    # 1. Load and Split
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    
    print(f"Splitting {len(raw_docs)} pages...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_docs)
    print(f"Total Chunks to process: {len(docs)}")

    # 2. Configure Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check Provider to determine Dimensions
    # Gemini = 768, OpenAI = 1536
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    dimension = 768 if provider == "gemini" else 1536
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Index '{PINECONE_INDEX_NAME}' with dimension {dimension}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    else:
        # Verify dimension matches
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        if index_info.dimension != dimension:
            print(f"CRITICAL ERROR: Index dimension is {index_info.dimension}, but model uses {dimension}.")
            print("Please delete the index in Pinecone console and run this script again.")
            return

    # 3. Batch Ingestion with Sleep (The 'Free Tier' Fix)
    embeddings = get_embedding_model()
    
    # We ingest in small batches (e.g., 20 chunks) and sleep to avoid 429 Errors
    batch_size = 20 
    total_docs = len(docs)
    
    print(f"Starting ingestion with batch size {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        print(f"Processing batch {i} to {i+len(batch)} / {total_docs}...")
        
        try:
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME
            )
            # SLEEP for 2 seconds between batches to respect Free Tier limits
            time.sleep(2) 
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            # If rate limit hit, wait longer and try continuing
            time.sleep(10)

    print("Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()
