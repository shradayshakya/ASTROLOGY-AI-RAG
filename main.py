import os
import sys
import re
from datetime import datetime
import streamlit as st

# Ensure src is importable when launching via Streamlit
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agent import create_agent_executor
from src.config import LANGCHAIN_API_KEY

st.set_page_config(page_title="Jyotish AI", page_icon="🕉️", layout="wide")

if not LANGCHAIN_API_KEY:
    st.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.", icon="⚠️")

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

    # Display previous chat messages (from Mongo history)
    try:
        history = agent_executor.get_session_history(st.session_state.session_id)
        for msg in history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content, unsafe_allow_html=True)
    except Exception:
        pass

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

    if st.button("End Session"):
        st.session_state.clear()
        st.rerun()
