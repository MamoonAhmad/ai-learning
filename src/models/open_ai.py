"""open_ai chat provider"""
from langchain_openai import ChatOpenAI


def get_chat_model(llm_model="gpt-4o"):
    """
        Gets Chat model for open AI
    """

    return ChatOpenAI(model=llm_model)
