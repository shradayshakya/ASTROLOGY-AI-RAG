import os
import sys
import re
from datetime import datetime
import streamlit as st
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agent import create_agent_executor
from src.config import LANGCHAIN_API_KEY
from src.auth import get_active_password, hash_password
from src.logging_utils import configure_logging, attach_console_handler, get_logger
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.ui.auth_gate import ensure_authenticated
from src.ui.session import init_session_state, render_session_form_and_create_agent, get_or_create_agent_executor
from src.ui.history import render_session_history
from src.ui.chat import handle_chat_interaction
from src.ui.session_end import render_end_session

configure_logging(logging.INFO)
attach_console_handler(logging.INFO)
app_logger = get_logger(__name__)
app_logger.info("Initializing Jyotish AI Streamlit app")
st.set_page_config(page_title="Jyotish AI", page_icon="🕉️", layout="wide")

if not LANGCHAIN_API_KEY:
    st.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.", icon="⚠️")
    app_logger.warning("LANGCHAIN_API_KEY not set; LangSmith tracing disabled.")

ensure_authenticated()

init_session_state()

st.title("🕉️ Jyotish AI")
st.caption("Agentic RAG for Vedic Astrology grounded in BPHS + real astronomical data.")

if not st.session_state.session_id:
    render_session_form_and_create_agent(app_logger)
else:
    st.success(f"Session started for **{st.session_state.session_id}**")
    agent_executor = get_or_create_agent_executor(st.session_state.session_id, app_logger)

    render_session_history(agent_executor, st.session_state.session_id, app_logger)

    handle_chat_interaction(agent_executor, st.session_state.session_id, st.session_state.user_profile, app_logger)

    render_end_session(app_logger)
