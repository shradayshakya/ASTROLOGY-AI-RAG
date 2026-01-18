from langchain_pinecone import PineconeVectorStore
from src.embedding_factory import get_embedding_model
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME 


def get_pinecone_retriever(top_k: int = 4):
    """Return a Pinecone retriever for BPHS search."""
    # Ensure client initialized (index must already exist via scripts/ingest.py)
    _ = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = get_embedding_model()
    vs = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": top_k})
