import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.logging_utils import get_logger, log_call
from langchain_aws import BedrockEmbeddings

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
    elif provider == "bedrock":
        region = os.getenv("AWS_REGION_NAME", "us-east-1")
        profile = os.getenv("AWS_PROFILE")
        logger.info(f"Using BedrockEmbeddings model=titan-embed-text-v2 region={region} profile={profile or 'env/default'}")
        # amazon.titan-embed-text-v2:0 outputs 1024-d vectors
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=region,
            credentials_profile_name=profile if profile else None,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def get_embedding_dimension() -> int:
    """Return the vector dimension for the configured embedding provider.
    openai → 1536, gemini → 768, bedrock (titan-v2) → 1024.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return 768
    if provider == "openai":
        return 1536
    if provider == "bedrock":
        return 1024
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}'")
