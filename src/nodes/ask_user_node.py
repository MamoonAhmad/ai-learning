
from langchain_core.messages import HumanMessage
from src.nodes.agent_state import AgentState


def ask_user_node(_: AgentState):
    """
    """

    user_message = input("Enter Your Answer or Question: ")

    return dict(
        conversation=[HumanMessage(content=user_message)]
    )
