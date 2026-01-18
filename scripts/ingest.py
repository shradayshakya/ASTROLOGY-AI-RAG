import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from src.embedding_factory import get_embedding_model, get_embedding_dimension
from pinecone import Pinecone, ServerlessSpec
from src.logging_utils import configure_logging, get_logger, log_call, log_operation

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME 

# Load environment variables from .env file
load_dotenv()
configure_logging()
_logger = get_logger(__name__)


@log_call
def ingest_data(pdf_path: str = "data/Brihat_Parashara_Hora_Shastra.pdf") -> Optional[None]:
    # 1. Load and Split
    if not os.path.exists(pdf_path):
        _logger.error(f"PDF not found at {pdf_path}")
        return

    _logger.info("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    with log_operation(_logger, "load_pdf"):
        raw_docs = loader.load()
    
    _logger.info(f"Splitting {len(raw_docs)} pages...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    with log_operation(_logger, "split_documents"):
        docs = text_splitter.split_documents(raw_docs)
    _logger.info(f"Total Chunks to process: {len(docs)}")

    # 2. Configure Pinecone
    with log_operation(_logger, "pinecone_client_init"):
        pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Determine index dimension via embedding_factory
    dimension = get_embedding_dimension()
    _logger.info(f"Embedding dimension selected: {dimension}")
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        _logger.info(f"Creating Index '{PINECONE_INDEX_NAME}' with dimension {dimension}...")
        with log_operation(_logger, "pinecone_create_index"):
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
            _logger.critical(
                f"Index dimension mismatch: got {index_info.dimension}, expected {dimension}."
            )
            _logger.error("Please delete the index in Pinecone console and run this script again.")
            return

    # 3. Batch Ingestion with Sleep (The 'Free Tier' Fix)
    with log_operation(_logger, "init_embeddings"):
        embeddings = get_embedding_model()
    
    # We ingest in small batches (e.g., 20 chunks) and sleep to avoid 429 Errors
    batch_size = 20 
    total_docs = len(docs)
    
    _logger.info(f"Starting ingestion with batch size {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        _logger.info(f"Processing batch {i} to {i+len(batch)} / {total_docs}...")
        
        try:
            with log_operation(_logger, "pinecone_upsert_batch"):
                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME
                )
            # SLEEP for 2 seconds between batches to respect Free Tier limits
            time.sleep(2) 
        except Exception as e:
            _logger.exception(f"Error on batch {i}: {e}")
            # If rate limit hit, wait longer and try continuing
            time.sleep(10)

    _logger.info("Ingestion Complete!")

if __name__ == "__main__":
    _logger.info("Running ingest_data from __main__")
    ingest_data()
