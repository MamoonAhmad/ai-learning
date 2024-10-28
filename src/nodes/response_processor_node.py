
from src.nodes.agent_state import AgentState


def response_processor_node(state: AgentState):

    if state['last_message_was_out_of_context']:
        print(
            "AI: I cannot answer this question. ",
            f"You have asked {state['out_of_context_question_counter']} out of context messages so far. ",
            "Please stay focused on the topic."
        )
