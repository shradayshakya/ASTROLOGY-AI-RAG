from langsmith import Client

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.tools import get_d10_chart, get_d9_chart, get_d1_chart, search_bphs
from src.config import MONGO_URI, MONGO_DB_NAME, MONGO_CHAT_HISTORY_COLLECTION



# Define context for astrology session (can be extended as needed)
@dataclass
class AstrologyContext:
    session_id: str

# Wrap astrology tools as agent tools
@tool
def d10_chart(runtime: ToolRuntime[AstrologyContext]) -> str:
    """Get D10 chart for the current session."""
    return get_d10_chart(runtime.context.session_id)

@tool
def d9_chart(runtime: ToolRuntime[AstrologyContext]) -> str:
    """Get D9 chart for the current session."""
    return get_d9_chart(runtime.context.session_id)

@tool
def d1_chart(runtime: ToolRuntime[AstrologyContext]) -> str:
    """Get D1 chart for the current session."""
    return get_d1_chart(runtime.context.session_id)

@tool
def bphs_search(runtime: ToolRuntime[AstrologyContext], query: str) -> str:
    """Search BPHS knowledge base."""
    return search_bphs(query)

def create_agent_executor(session_id: str):
    """Create an agent with tools and chat history bound to session_id."""
    import os
    prompt_repo_path = os.environ.get("JYOTISH_AI_PROMPT_REPO", "shradayshakya/jyotish-ai")
    client = Client()
    try:
        prompt_obj = client.get_prompt(prompt_repo_path)
        system_prompt = prompt_obj.messages[0]["content"] if prompt_obj.messages else "You are Jyotish AI."
    except Exception:
        system_prompt = (
            "You are an expert Vedic Astrologer named 'Jyotish AI'. Your knowledge comes only from tools and BPHS.\n"
            "Strictly use Vedic (Sidereal) principles; refuse non-astrology, politics, stocks, gambling."
        )

    from src.llm_factory import get_chat_model
    model = get_chat_model()
    agent = create_agent(
        model,
        tools=[d10_chart, d9_chart, d1_chart, bphs_search],
        context_schema=AstrologyContext,
        system_prompt=system_prompt,
    )

    # Optionally wrap with chat history if needed
    agent_with_history = RunnableWithMessageHistory(
        agent,
        lambda sid: MongoDBChatMessageHistory(
            session_id=sid,
            connection_string=MONGO_URI,
            database_name=MONGO_DB_NAME,
            collection_name=MONGO_CHAT_HISTORY_COLLECTION,
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_history
