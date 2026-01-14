from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.llm_factory import get_chat_model
from src.tools import get_d10_chart, get_d9_chart, get_d9_chart_info, get_d1_chart, search_bphs
from src.config import MONGO_URI, MONGO_DB_NAME, MONGO_CHAT_HISTORY_COLLECTION


def create_agent_executor(session_id: str):
    """Create an AgentExecutor with tools and chat history bound to session_id."""
    llm = get_chat_model()
    tools = [get_d10_chart, get_d9_chart, get_d9_chart_info, get_d1_chart, search_bphs]

    # Pull prompt from LangChain Hub (replace with your repo when ready)
    try:
        prompt = hub.pull("hwchase17/react-chat")
        # Example: prompt = hub.pull("your-username/vedic-rag-agent")
    except Exception:
        from langchain.prompts import PromptTemplate
        template = (
            "You are a helpful Vedic Astrologer assistant. Use tools to fetch charts "
            "and search BPHS. Render SVG by including it directly in the final answer.\n\n"
            "Tools: {tools}\n\n"
            "Question: {input}\n"
            "{agent_scratchpad}"
        )
        prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    agent_with_history = RunnableWithMessageHistory(
        executor,
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
