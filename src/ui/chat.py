import re
import streamlit as st

from langchain_core.messages import HumanMessage


def handle_chat_interaction(agent_executor, session_id, user_profile, app_logger):
    # Chat input
    prompt = st.chat_input("Ask about career (D10), marriage (D9), or health (D1)...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        profile = user_profile
        composed = (
            "User Profile:\n"
            f"DOB: {profile['dob']}\nTime: {profile['tob']}\nCity: {profile['city']}\n\n"
            "User Input:\n"
            f"{prompt}\n\n"
        )
        with st.chat_message("assistant"):
            with st.spinner("Consulting charts and classical texts..."):
                try:
                    resp = agent_executor.invoke(
                        {"messages": [HumanMessage(content=composed)]},
                        {"configurable": {"session_id": session_id}},
                    )

                    def _block_text(content):
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            parts = []
                            for b in content:
                                if isinstance(b, dict):
                                    t = b.get("text")
                                    if isinstance(t, str):
                                        parts.append(t)
                            return "\n".join(parts) if parts else str(content)
                        return str(content)

                    def _extract_output(r):
                        if isinstance(r, dict):
                            out = r.get("output")
                            if isinstance(out, str) and out:
                                return out
                            msgs = r.get("messages")
                            if isinstance(msgs, list) and msgs:
                                try:
                                    from langchain_core.messages import AIMessage as ai_message_cls
                                except Exception:
                                    ai_message_cls = None
                                for m in reversed(msgs):
                                    if ai_message_cls is not None and isinstance(m, ai_message_cls):
                                        return _block_text(m.content)
                                    if isinstance(m, dict) and (m.get("type") == "ai" or m.get("role") == "assistant"):
                                        return _block_text(m.get("content"))
                            return str(r)
                        if hasattr(r, "content"):
                            return _block_text(getattr(r, "content"))
                        return str(r)

                    def _sanitize_and_capture_thinking(text):
                        if not isinstance(text, str):
                            return "", []
                        pattern = re.compile(r"<thinking>(.*?)</thinking>", flags=re.DOTALL | re.IGNORECASE)
                        segments = [m.strip() for m in pattern.findall(text)]
                        cleaned = pattern.sub("", text)
                        return cleaned.strip(), segments

                    output = _extract_output(resp)
                    sanitized_out, thinking_notes_out = _sanitize_and_capture_thinking(output)
                    st.markdown(sanitized_out or "Sorry, I encountered an error.", unsafe_allow_html=True)
                    if thinking_notes_out:
                        with st.expander("Show assistant notes", expanded=False):
                            st.markdown("**Hidden reasoning**")
                            for seg in thinking_notes_out:
                                st.code(seg)
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
