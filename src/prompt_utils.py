from typing import Optional
from langsmith import Client
from src.logging_utils import get_logger, log_call
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

logger = get_logger(__name__)


@log_call
def get_prompt_content(prompt_name: str) -> str:
    """Get a prompt's text content from LangSmith.
    Handles both ChatPromptTemplate (messages) and PromptTemplate (template).
    Fallbacks to string serialization when shape is unknown.
    """
    logger.info(f"Pulling prompt '{prompt_name}' from LangSmith")
    client = Client()
    prompt = client.pull_prompt(prompt_name)

    # Chat prompt: extract first message template/content
    try:
        if isinstance(prompt, ChatPromptTemplate):
            messages = getattr(prompt, "messages", [])
            if messages:
                first = messages[0]
                # Many ChatPromptTemplate messages wrap a PromptTemplate in .prompt
                if hasattr(first, "prompt") and hasattr(first.prompt, "template"):
                    return first.prompt.template
                # Some messages carry .template directly
                if hasattr(first, "template"):
                    return first.template
                # Or a content field (rare)
                if hasattr(first, "content"):
                    return first.content
    except Exception as e:
        logger.warning(f"Failed to parse ChatPromptTemplate: {e}")

    # Text prompt: standard template
    try:
        if isinstance(prompt, PromptTemplate):
            return prompt.template
    except Exception as e:
        logger.warning(f"Failed to parse PromptTemplate: {e}")

    # Generic fallback: try direct .template
    if hasattr(prompt, "template"):
        try:
            return prompt.template  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Failed to access .template attribute: {e}")

    # Last resort: string representation
    return str(prompt)
