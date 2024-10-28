"""Exports functions to get models for a specific provider"""
from .open_ai import get_chat_model as get_open_ai_chat_model
from .groq import get_chat_model as get_groq_chat_model

# can be set in env
AI_PROVIDER = "open_ai"


def get_chat_model():
    """
        Get the chat model for a provider.
    """

    if AI_PROVIDER == "open_ai":
        return get_open_ai_chat_model()

    elif AI_PROVIDER == "groq":
        return get_groq_chat_model()
