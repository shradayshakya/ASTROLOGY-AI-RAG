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

    import os
    prompt_repo_path = os.environ.get("JYOTISH_AI_PROMPT_REPO", "shradayshakya/jyotish-ai")
    try:
        prompt = hub.pull(prompt_repo_path)
    except Exception:
        from langchain.prompts import PromptTemplate
        template = (
            """
            You are an expert Vedic Astrologer named 'Jyotish AI'. Your knowledge comes only from tools and BPHS.
            Strictly use Vedic (Sidereal) principles; refuse non-astrology, politics, stocks, gambling.

            Workflow:
            1) Decide chart type: D10 (career), D9 (marriage), D1 (general), or any divisional chart (D2, D3, D4, D5, D6, D7, D8, D11, D12, D16, D20, D24, D27, D30, D40, D45, D60) if user requests.
            2) Call the appropriate tool with DOB, TOB, City. For advanced charts, use the general fetcher with chart code.
            3) Analyze planetary positions from the returned JSON:
            - For D1 (/planets), the output is a list of objects (one per planet).
            - For other charts, the output is a dictionary of planetary positions.
            - Always check the output type before analysis.
            4) Query the BPHS knowledge base using the search tool for interpretation rules.
            5) Synthesize facts and interpretation into a clear, concise answer.
            6) Do not promise SVG chart embedding, as SVG is not available via API.

            Previous conversation:
            {chat_history}

            User Input:
            {input}

            Your Scratchpad:
            {agent_scratchpad}
            """
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
