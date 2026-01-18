from langchain import OpenAI
import streamlit as st
from src.agent import create_agent
from src.logging_utils import configure_logging, get_logger, log_call

@log_call
def main():
    configure_logging()
    logger = get_logger(__name__)
    st.title("Jyotish AI Chatbot")
    st.write("Welcome to the Jyotish AI Chatbot! Ask your questions below:")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        logger.info("Initializing chat history in session state")
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("You:", "")

    if st.button("Send") and user_input:
        # Append user input to chat history
        logger.info("Received user input; creating agent and generating response")
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Create the agent and get the response
        try:
            agent = create_agent()
            response = agent.run(user_input)
        except Exception as e:
            logger.exception(f"Agent run failed: {e}")
            response = "Sorry, something went wrong while generating a response."

        # Append assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display chat history
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Assistant:** {chat['content']}")
        logger.info("Chat history rendered")

if __name__ == "__main__":
    main()