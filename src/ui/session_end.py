import streamlit as st


def render_end_session(app_logger):
    if st.button("End Session"):
        app_logger.info("Ending session and clearing state")
        st.session_state.clear()
        st.rerun()
