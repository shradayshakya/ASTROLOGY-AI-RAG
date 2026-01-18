
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from src.logging_utils import configure_logging, get_logger, log_call, log_operation

"""
Run: `python scripts/setup_prompts.py`
"""
def setup_langchain_hub_prompt():
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Preparing to push prompt to LangSmith")
    
    
    @log_call
    def _push_prompt():
    import os
    prompt_repo_path = os.environ.get("JYOTISH_AI_PROMPT_REPO", "jyotish-ai")

    template = """
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
    prompt = ChatPromptTemplate.from_template(template)
    logger.info(f"Pushing prompt to LangSmith: {prompt_repo_path}")
    client = Client()
    try:
        with log_operation(logger, "push_prompt"):
            client.push_prompt(prompt_repo_path, object=prompt)
        logger.info("Prompt pushed successfully.")
    except Exception as e:
        logger.exception(f"Failed to push prompt: {e}")
    
    _push_prompt()


if __name__ == "__main__":
    setup_langchain_hub_prompt()
