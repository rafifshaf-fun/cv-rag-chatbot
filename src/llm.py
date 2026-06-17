"""
LLM setup — primary model, fallback, and chat-history store.
"""

import os

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from config import PRIMARY_MODEL, FALLBACK_MODEL, LLM_TEMPERATURE

# ── In-memory session store ────────────────────────────────────────────
_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (and cache) a ChatMessageHistory for the given session."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def make_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=LLM_TEMPERATURE,
    )


def get_llm_with_fallback():
    """
    Return an LLM that automatically falls back from PRIMARY_MODEL
    to FALLBACK_MODEL on rate-limit / availability errors.
    """
    primary = make_llm(PRIMARY_MODEL)
    fallback = make_llm(FALLBACK_MODEL)
    return primary.with_fallbacks(
        [fallback],
        exceptions_to_handle=(ResourceExhausted, ServiceUnavailable),
    )
