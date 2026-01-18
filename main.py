import os
import sys
import re
from datetime import datetime
import streamlit as st
import logging

# Ensure src is importable when launching via Streamlit
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agent import create_agent_executor
from src.config import LANGCHAIN_API_KEY
from src.logging_utils import configure_logging, attach_console_handler, get_logger

configure_logging(logging.INFO)
attach_console_handler(logging.INFO)
app_logger = get_logger(__name__)
app_logger.info("Initializing Jyotish AI Streamlit app")
st.set_page_config(page_title="Jyotish AI", page_icon="🕉️", layout="wide")

if not LANGCHAIN_API_KEY:
    st.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.", icon="⚠️")
    app_logger.warning("LANGCHAIN_API_KEY not set; LangSmith tracing disabled.")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

st.title("🕉️ Jyotish AI")
st.caption("Agentic RAG for Vedic Astrology grounded in BPHS + real astronomical data.")

if not st.session_state.session_id:
    st.info("Enter your details to begin.")
    with st.form("user_profile_form"):
        email = st.text_input("Email (Session ID)")
        dob = st.date_input("Date of Birth", value=datetime(1990, 1, 1))
        tob = st.time_input("Time of Birth")
        city = st.text_input("City of Birth (e.g., 'New Delhi, India')")
        submitted = st.form_submit_button("Start Session")
        if submitted:
            if email and dob and tob and city:
                app_logger.info("Starting session and creating agent executor")
                st.session_state.session_id = email
                st.session_state.user_profile = {
                    "dob": dob.strftime("%Y-%m-%d"),
                    "tob": tob.strftime("%H:%M"),
                    "city": city,
                }
                st.session_state.agent_executor = create_agent_executor(st.session_state.session_id)
                st.rerun()
            else:
                st.error("Please fill in all fields.")
else:
    st.success(f"Session started for **{st.session_state.session_id}**")
    agent_executor = st.session_state.agent_executor
    if agent_executor is None:
        app_logger.warning("agent_executor missing in session; recreating")
        try:
            agent_executor = create_agent_executor(st.session_state.session_id)
            st.session_state.agent_executor = agent_executor
            app_logger.info("agent_executor recreated successfully")
        except Exception as e:
            app_logger.exception(f"Failed to create agent_executor: {e}")
            st.error("Unable to initialize the agent. Please try restarting the session.")
            st.stop()

    # Display previous chat messages (from Mongo history)
    try:
        history = agent_executor.get_session_history(st.session_state.session_id)
        for msg in history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content, unsafe_allow_html=True)
    except Exception as e:
        app_logger.warning(f"Failed to load chat history: {e}")

    # Chat input
    prompt = st.chat_input("Ask about career (D10), marriage (D9), or health (D1)...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        profile = st.session_state.user_profile
        composed = (
            f"DOB: {profile['dob']}\nTime: {profile['tob']}\nCity: {profile['city']}\n\n"
            f"Question: {prompt}"
        )
        with st.chat_message("assistant"):
            with st.spinner("Consulting charts and classical texts..."):
                try:
                    resp = agent_executor.invoke(
                        {"messages": [{"role": "user", "content": composed}]},
                        {"configurable": {"session_id": st.session_state.session_id}},
                    )
                    output = resp.get("output") if isinstance(resp, dict) else resp
                    st.markdown(output or "Sorry, I encountered an error.", unsafe_allow_html=True)
                except Exception as e:
                    msg = str(e)
                    retry_secs = None
                    m = re.search(r"retry in\s*([\d\.]+)s", msg, re.IGNORECASE)
                    if m:
                        try:
                            retry_secs = int(float(m.group(1)))
                        except Exception:
                            retry_secs = None
                    friendly = "The Jyotish AI is meditating now, come back later for deeper insights."
                    if retry_secs:
                        friendly += " Try again tomorrow."
                    st.warning(friendly)
                    app_logger.exception(f"Agent invocation failed: {e}")

    if st.button("End Session"):
        app_logger.info("Ending session and clearing state")
        st.session_state.clear()
        st.rerun()
