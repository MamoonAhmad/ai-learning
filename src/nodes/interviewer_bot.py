from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage

from src.models import get_chat_model

from src.nodes.agent_state import AgentState
from src.nodes.ask_user_node import ask_user_node
from src.nodes.general_context_node import general_context_node
from src.nodes.tech_analyzer_node import tech_analyzer_node
from src.nodes.response_processor_node import response_processor_node


class InterviewerBot:

    def __init__(self, question):

        self.question = question
        graph = StateGraph(AgentState)
        graph.add_node("ask_user", ask_user_node)
        graph.add_node("general_context", general_context_node)
        graph.add_node("tech_analyzer", tech_analyzer_node)
        graph.add_node("print_question", self.ask_question)
        graph.add_node("response_processor_node", response_processor_node)

        graph.add_edge("print_question", "ask_user")
        graph.add_edge("ask_user", "general_context")
        graph.add_conditional_edges(
            "general_context",
            self.was_last_question_out_of_context,
            {True: "response_processor_node", False: "tech_analyzer"}
        )
        graph.add_edge("tech_analyzer", "response_processor_node")
        graph.add_edge("response_processor_node", "ask_user")
        graph.set_entry_point("print_question")
        self.graph = graph.compile()

        self.model = get_chat_model()

    def was_last_question_out_of_context(self, state: AgentState):

        return bool(state["last_message_was_out_of_context"])

    def ask_question(self, _: AgentState):

        print(self.question)

        return dict(
            conversation=[SystemMessage(
                content=f"Original Question: {self.question}"
            )]
        )
