import json
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def render_session_history(agent_executor, session_id, app_logger):
    # Display previous chat messages (from Mongo history)
    try:
        history = agent_executor.get_session_history(session_id)

        def _msg_block_text(content):
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        t = b.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                    elif isinstance(b, str):
                        parts.append(b)
                return "\n".join(parts) if parts else str(content)
            return str(content)

        def _sanitize_and_capture_thinking(text):
            if not isinstance(text, str):
                return "", []
            import re
            pattern = re.compile(r"<thinking>(.*?)</thinking>", flags=re.DOTALL | re.IGNORECASE)
            segments = [m.strip() for m in pattern.findall(text)]
            cleaned = pattern.sub("", text)
            return cleaned.strip(), segments

        tool_notes_buffer = []
        for msg in history.messages:
            # Skip rendering ToolMessage bubble; buffer for next AIMessage expander
            if isinstance(msg, ToolMessage):
                tool_notes_buffer.append(_msg_block_text(getattr(msg, "content", "")))
                continue
            # Human/user message
            if isinstance(msg, HumanMessage):
                text = _msg_block_text(getattr(msg, "content", ""))
                if text:
                    with st.chat_message("user"):
                        st.markdown(text, unsafe_allow_html=True)
                continue
            # Assistant message with hidden notes
            if isinstance(msg, AIMessage):
                raw = _msg_block_text(getattr(msg, "content", ""))
                sanitized, thinking_notes = _sanitize_and_capture_thinking(raw)
                with st.chat_message("assistant"):
                    if sanitized:
                        st.markdown(sanitized, unsafe_allow_html=True)
                    if thinking_notes or tool_notes_buffer:
                        with st.expander("Show assistant notes", expanded=False):
                            if thinking_notes:
                                st.markdown("**Hidden reasoning**")
                                for seg in thinking_notes:
                                    st.code(seg)
                            if tool_notes_buffer:
                                st.markdown("**Tool outputs**")
                                for note in tool_notes_buffer:
                                    try:
                                        parsed = json.loads(note)
                                        st.json(parsed)
                                    except Exception:
                                        st.code(str(note))
                    tool_notes_buffer = []
                continue
            # Fallback for unknown message types
            with st.chat_message(getattr(msg, "type", "assistant")):
                st.markdown(_msg_block_text(getattr(msg, "content", "")), unsafe_allow_html=True)
    except Exception as e:
        app_logger.warning(f"Failed to load chat history: {e}")
