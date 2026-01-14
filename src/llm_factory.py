from src.config import LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY, AWS_REGION_NAME
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrock


def get_chat_model():
    """
    Factory Method to return a ChatModel based on LLM_PROVIDER.
    Supports: OpenAI GPT-4o, Google Gemini 1.5 Pro, AWS Bedrock Claude 3 Sonnet.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    if provider in ("google_genai", "gemini"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

    if provider == "bedrock":
        return ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=AWS_REGION_NAME,
            model_kwargs={"temperature": 0},
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}. Use 'openai', 'google_genai', or 'bedrock'."
    )
