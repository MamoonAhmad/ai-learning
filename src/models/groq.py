"""groq AI provider"""
from langchain_groq import ChatGroq


def get_chat_model():
    """
        Gets Chat Model for Groq
    """

    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
