import streamlit as st

from src.auth import get_active_password, hash_password


def ensure_authenticated():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 Access Restricted")
        st.write("Please enter the password to access the Vedic Astrologer.")

        password_input = st.text_input("Password", type="password")

        if st.button("Login"):
            active_password_hash = get_active_password()
            if hash_password(password_input or "") == active_password_hash:
                st.session_state["authenticated"] = True
                st.success("Access Granted!")
                st.rerun()  # Reload the app to show the actual content
            else:
                st.error("Incorrect Password. Please try again.")

        st.stop()
