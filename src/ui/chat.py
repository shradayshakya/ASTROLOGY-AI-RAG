from langchain import OpenAI
import streamlit as st
from src.agent import create_agent

def main():
    st.title("Vedic Astrology Chatbot")
    st.write("Welcome to the Vedic Astrology Chatbot! Ask your questions below:")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if user_input:
            # Append user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Create the agent and get the response
            agent = create_agent()
            response = agent.run(user_input)

            # Append assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display chat history
            for chat in st.session_state.chat_history:
                if chat['role'] == 'user':
                    st.markdown(f"**You:** {chat['content']}")
                else:
                    st.markdown(f"**Assistant:** {chat['content']}")

if __name__ == "__main__":
    main()