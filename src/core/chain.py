from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os

class VedicAstrologyChain:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _initialize_llm(self):
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        if llm_provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Add other LLM providers as needed
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def _create_prompt_template(self):
        return PromptTemplate(
            input_variables=["user_input"],
            template="You are a Vedic astrologer. Answer the following question: {user_input}"
        )

    def run(self, user_input):
        return self.chain.run(user_input)

    def initialize_agent(self, tools):
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent

    def add_tool(self, tool_name, tool_function):
        return Tool(
            name=tool_name,
            func=tool_function,
            description=f"Use this tool to {tool_name}."
        )