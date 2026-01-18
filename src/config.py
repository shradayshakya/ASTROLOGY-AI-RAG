# LangChain Hub Prompt Repo
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# LangSmith / LangChain Hub
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "JYOTISH-AI")
JYOTISH_AI_PROMPT_REPO = os.getenv("JYOTISH_AI_PROMPT_REPO", "your-username/jyotish-ai")

# LLM Provider: openai | google_genai | bedrock
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# OpenAI / Google / Bedrock
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")


# Pinecone & Embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jyotish-ai-index")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "jyotish_ai_cache")
MONGO_CHAT_HISTORY_COLLECTION = os.getenv("MONGO_CHAT_HISTORY_COLLECTION", "chat_history")
MONGO_API_CACHE_COLLECTION = os.getenv("MONGO_API_CACHE_COLLECTION", "api_cache")

# FreeAstrologyAPI
FREE_ASTROLOGY_API_KEY = os.getenv("FREE_ASTROLOGY_API_KEY")
ASTRO_OBSERVATION_POINT = os.getenv("ASTRO_OBSERVATION_POINT", "topocentric")
ASTRO_AYANAMSHA = os.getenv("ASTRO_AYANAMSHA", "lahiri")
ASTRO_LANGUAGE = os.getenv("ASTRO_LANGUAGE", "en")
