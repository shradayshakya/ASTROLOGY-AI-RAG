from src.config import LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY, AWS_REGION_NAME
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrock
from src.logging_utils import get_logger, log_call

logger = get_logger(__name__)


@log_call
def get_chat_model():
    """
    Factory Method to return a ChatModel based on LLM_PROVIDER.
    Supports: OpenAI GPT-4.1-nano, Google Gemini 3 Pro Preview, AWS Bedrock Nova Lite.
    """
    provider = LLM_PROVIDER.lower()
    logger.info(f"Selecting chat model provider: {provider}")

    if provider == "openai":
        model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY)
        return model.with_config(
            {
                "run_name": "LLM • OpenAI gpt-4.1-nano",
                "tags": ["llm", "provider:openai"],
                "metadata": {"model": "gpt-4.1-nano", "temperature": 0},
            }
        )

    if provider in ("google_genai", "gemini"):
        model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=GOOGLE_API_KEY)
        return model.with_config(
            {
                "run_name": "LLM • Google Gemini 3 Pro (preview)",
                "tags": ["llm", "provider:google"],
                "metadata": {"model": "gemini-3-pro-preview"},
            }
        )

    if provider == "bedrock":
        model = ChatBedrock(
            model_id="amazon.nova-lite-v1:0",
            region_name=AWS_REGION_NAME,
            model_kwargs={"temperature": 0},
        )
        return model.with_config(
            {
                "run_name": "LLM • Bedrock amazon.nova-lite-v1",
                "tags": ["llm", "provider:bedrock"],
                "metadata": {"model": "amazon.nova-lite-v1:0", "temperature": 0},
            }
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}. Use 'openai', 'google_genai', or 'bedrock'."
    )
