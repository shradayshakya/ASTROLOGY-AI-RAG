import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.logging_utils import get_logger, log_call

logger = get_logger(__name__)

def get_embedding_model():
    """
    Returns an embedding model instance based on the EMBEDDING_PROVIDER environment variable.
    Defaults to Gemini if not specified.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    logger.info(f"Selecting embedding provider: {provider}")
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
