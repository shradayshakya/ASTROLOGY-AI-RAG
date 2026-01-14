from langchain import hub
from langchain.prompts import PromptTemplate

"""
Run: `python scripts/setup_prompts.py`
Prereqs:
- LANGCHAIN_HUB_API_KEY set
- `langchain hub login` done in your shell
- Replace `your-username` below with your actual hub username
"""

def setup_langchain_hub_prompt():
    prompt_repo_path = "your-username/vedic-rag-agent"  # TODO: set correctly

    template = """
You are an expert Vedic Astrologer named 'Vedic-RAG'. Your knowledge comes only from tools and BPHS.
Strictly use Vedic (Sidereal) principles; refuse non-astrology, politics, stocks, gambling.

Workflow:
1) Decide chart type: D10 (career), D9 (marriage), D1 (general).
2) Call the appropriate tool with DOB, TOB, City.
3) Analyze planetary positions from returned JSON.
4) Query internal knowledge base (BPHS) with a targeted search.
5) Synthesize facts + interpretation into a clear answer.
6) Include the SVG chart at the end of your answer (embed directly).

Previous conversation:
{chat_history}

User Input:
{input}

Your Scratchpad:
{agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(template)
    print(f"Pushing prompt to Hub: {prompt_repo_path}")
    try:
        hub.push(prompt_repo_path, prompt)
        print("Prompt pushed successfully.")
    except Exception as e:
        print("Failed to push prompt:", e)
        print("Ensure LANGCHAIN_HUB_API_KEY is set and 'langchain hub login' succeeded.")


if __name__ == "__main__":
    setup_langchain_hub_prompt()
