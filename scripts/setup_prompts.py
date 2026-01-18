import os
import sys
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


# Ensure the project root is on sys.path so 'src.*' imports work when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logging_utils import configure_logging, get_logger

from src.config import JYOTISH_AI_PROMPT_REPO 

# Load environment variables from .env file
load_dotenv()
configure_logging()
_logger = get_logger(__name__)


"""
Run: `python scripts/setup_prompts.py`
"""
def setup_langchain_hub_prompt():
    prompt_repo_path = JYOTISH_AI_PROMPT_REPO

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
4) DO NOT call any BPHS search tool. Instead, use the BPHS passages injected inline under the heading "Brihat Parashara Hora Shastra (BPHS)" within the user's message. Ground your interpretation in those passages.
5) Synthesize facts and interpretation into a clear, concise answer. Cite relevant BPHS lines (quote brief snippets) when appropriate.
6) Do not promise SVG chart embedding, as SVG is not available via API.

Previous conversation:
{chat_history}

User Input:
{input}

Astrology API Responses:
{agent_scratchpad}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print(f"Pushing prompt to LangSmith: {prompt_repo_path}")
    client = Client()
    try:
        client.push_prompt(prompt_repo_path, object=prompt)
        print("Prompt pushed successfully.")
    except Exception as e:
        print("Failed to push prompt:", e)


if __name__ == "__main__":
    setup_langchain_hub_prompt()
