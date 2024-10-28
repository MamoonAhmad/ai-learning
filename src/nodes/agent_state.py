
from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, AIMessage


class AgentState(TypedDict):
    """
    """
    times_user_was_rude: int
    # Messages between AI and user
    conversation: Annotated[List[AnyMessage], operator.add]
    out_of_context_question_counter: int
    last_message_was_out_of_context: bool
    last_ai_response: AIMessage


def get_initial_state():

    return AgentState(
        conversation=[],
        last_ai_response=None,
        last_message_was_out_of_context=False,
        out_of_context_question_counter=0,
        times_user_was_rude=0
    )
