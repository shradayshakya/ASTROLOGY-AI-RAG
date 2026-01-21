from datetime import datetime
import streamlit as st

from src.agent import create_agent_executor


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None


def render_session_form_and_create_agent(app_logger):
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


def get_or_create_agent_executor(session_id, app_logger):
    agent_executor = st.session_state.agent_executor
    if agent_executor is None:
        app_logger.warning("agent_executor missing in session; recreating")
        try:
            agent_executor = create_agent_executor(session_id)
            st.session_state.agent_executor = agent_executor
            app_logger.info("agent_executor recreated successfully")
        except Exception as e:
            app_logger.exception(f"Failed to create agent_executor: {e}")
            st.error("Unable to initialize the agent. Please try restarting the session.")
            st.stop()
    return agent_executor
