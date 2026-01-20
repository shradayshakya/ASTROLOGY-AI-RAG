import re
from langchain import OpenAI
import streamlit as st
from src.agent import create_agent
from src.logging_utils import configure_logging, get_logger, log_call

# Import LangChain message classes (modern path). If unavailable, fall back to None.
try:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
except Exception:
    HumanMessage = None
    AIMessage = None
    ToolMessage = None


def _sanitize_and_capture_thinking(text: str):
    """Return (sanitized_text, thinking_segments) where thinking segments are removed.

    thinking_segments list contains the raw inner text of <thinking>...</thinking> blocks.
    """
    if not isinstance(text, str):
        return "", []
    pattern = re.compile(r"<thinking>(.*?)</thinking>", flags=re.DOTALL | re.IGNORECASE)
    segments = [m.strip() for m in pattern.findall(text)]
    cleaned = pattern.sub("", text)
    return cleaned.strip(), segments


def _get_text_content(message) -> str:
    """Extract textual content from LangChain message variants.

    Handles content that may be a string or a list of blocks and sanitizes it.
    """
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content_str = _extract_text_from_blocks(content)
    else:
        content_str = content if isinstance(content, str) else ""
    sanitized, _notes = _sanitize_and_capture_thinking(content_str)
    return sanitized


def _get_ai_text_and_notes(message):
    """Extract sanitized AI text and hidden notes (<thinking> segments)."""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content_str = _extract_text_from_blocks(content)
    else:
        content_str = content if isinstance(content, str) else ""
    sanitized, thinking_notes = _sanitize_and_capture_thinking(content_str)
    return sanitized, thinking_notes


def _extract_text_from_blocks(blocks) -> str:
    """Extract text from a list of content blocks in LC messages."""
    parts = []
    for block in blocks:
        if isinstance(block, dict):
            text_val = block.get("text")
            if block.get("type") == "text" and isinstance(text_val, str):
                parts.append(text_val)
            elif isinstance(block.get("content"), str):
                parts.append(block["content"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def render_langchain_history(messages):
    """Render LangChain messages with sanitized bubbles and expandable notes."""
    typed = bool(HumanMessage and AIMessage and ToolMessage)
    if not typed:
        _render_untyped_history(messages)
        return
    _render_typed_history(messages)


def _render_untyped_history(messages):
    for msg in messages:
        text = _get_text_content(msg)
        if text:
            st.markdown(text)


def _render_typed_history(messages):
    tool_notes_buffer = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_notes_buffer.append(getattr(msg, "content", None))
            continue
        if isinstance(msg, HumanMessage):
            text = _get_text_content(msg)
            if text:
                st.markdown(f"**You:** {text}")
            continue
        if isinstance(msg, AIMessage):
            text, thinking_notes = _get_ai_text_and_notes(msg)
            _render_ai_with_notes(text, thinking_notes, tool_notes_buffer)
            tool_notes_buffer = []
            continue
        # Unknown type fallback
        text = _get_text_content(msg)
        if text:
            st.markdown(text)


def _render_ai_with_notes(text: str, thinking_notes, tool_notes):
    """Render an assistant bubble and optional notes in an expander."""
    if text:
        st.markdown(f"**Assistant:** {text}")
    if not (thinking_notes or tool_notes):
        return
    with st.expander("Show assistant notes", expanded=False):
        _render_thinking_notes(thinking_notes)
        _render_tool_notes(tool_notes)


def _render_thinking_notes(notes):
    if not notes:
        return
    st.markdown("**Hidden reasoning**")
    for seg in notes:
        st.code(seg)


def _render_tool_notes(notes):
    if not notes:
        return
    st.markdown("**Tool outputs**")
    for note in notes:
        if isinstance(note, (dict, list)):
            st.json(note)
        else:
            st.code(str(note))


def render_simple_history(chat_history):
    """Render simple role/content dict history with sanitization and optional notes expander."""
    for chat in chat_history:
        role = chat.get('role')
        content = chat.get('content', '')
        if role == 'assistant':
            sanitized, thinking_notes = _sanitize_and_capture_thinking(content)
            if sanitized:
                st.markdown(f"**Assistant:** {sanitized}")
            if thinking_notes:
                with st.expander("Show assistant notes", expanded=False):
                    st.markdown("**Hidden reasoning**")
                    for seg in thinking_notes:
                        st.code(seg)
        elif role == 'user':
            st.markdown(f"**You:** {content}")

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

        # Prefer rendering LangChain message history if present in session
        # e.g., st.session_state.lc_history populated by the agent/memory
        lc_history = st.session_state.get("lc_history")
        if lc_history:
            render_langchain_history(lc_history)
        else:
            render_simple_history(st.session_state.chat_history)