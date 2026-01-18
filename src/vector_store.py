from langchain_pinecone import PineconeVectorStore
from src.embedding_factory import get_embedding_model
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME 
from src.logging_utils import get_logger, log_call

logger = get_logger(__name__)


@log_call
def get_pinecone_retriever(top_k: int = 4):
    """Return a Pinecone retriever for BPHS search."""
    # Ensure client initialized (index must already exist via scripts/ingest.py)
    logger.info("Initializing Pinecone client for retriever")
    _ = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = get_embedding_model()
    # Use the same namespace as ingestion to retrieve documents
    vs = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace="bphs")
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    logger.info(f"Retriever initialized with top_k={top_k}")
    return retriever
