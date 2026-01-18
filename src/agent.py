from langsmith import Client

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.prompt_utils import get_prompt_content

from src.tools import (
    get_d10_chart, get_d9_chart, get_d1_chart, get_d2_chart, get_d7_chart, get_d24_chart,
    get_specific_varga_chart, search_bphs
)
from src.config import MONGO_URI, MONGO_DB_NAME, MONGO_CHAT_HISTORY_COLLECTION
from src.logging_utils import get_logger, log_call

# Define context for astrology session (can be extended as needed)
@dataclass
class AstrologyContext:
    session_id: str


logger = get_logger(__name__)

@log_call
def create_agent_executor(session_id: str):
    """Create an agent with tools and chat history bound to session_id."""
    import os
    prompt_repo_path = os.environ.get("JYOTISH_AI_PROMPT_REPO", "jyotish-ai")
    try:
        logger.info("Fetching system prompt via prompt_utils.get_prompt_content")
        system_prompt = get_prompt_content(prompt_repo_path)
    except Exception as e:
        logger.warning(f"Failed to fetch prompt from LangSmith: {e}. Falling back to default.")
        system_prompt = (
            "You are an expert Vedic Astrologer named 'Jyotish AI'. Your knowledge comes only from tools and BPHS.\n"
            "Strictly use Vedic (Sidereal) principles; refuse non-astrology, politics, stocks, gambling."
        )

    from src.llm_factory import get_chat_model
    logger.info("Creating chat model via factory")
    model = get_chat_model()
    agent = create_agent(
        model,
        tools=[
            get_d10_chart,
            get_d9_chart,
            get_d1_chart,
            get_d2_chart,
            get_d7_chart,
            get_d24_chart,
            get_specific_varga_chart,
            # Pinecone BPHS search tool for RAG context
            search_bphs,
        ],
        context_schema=AstrologyContext,
        system_prompt=system_prompt,
    )
    logger.info("Agent created with tools")

    # Optionally wrap with chat history if needed
    agent_with_history = RunnableWithMessageHistory(
        agent,
        lambda sid: MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=MONGO_URI,
            database_name=MONGO_DB_NAME,
            collection_name=MONGO_CHAT_HISTORY_COLLECTION,
        ),
        input_messages_key="messages",
        history_messages_key="chat_history",
    )
    logger.info("Agent wrapped with MongoDB chat history")

    return agent_with_history
