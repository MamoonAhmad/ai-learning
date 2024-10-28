from unittest import TestCase
from src.nodes.general_context_node import general_context_node
from src.nodes.agent_state import AgentState, get_initial_state
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def initiate_state(partial_state):
    state = get_initial_state()
    state.update(partial_state)
    return state


class general_context_node_tests(TestCase):

    def test_user_response_ask_for_answer_directly(self):

        state = initiate_state(dict(conversation=[
            SystemMessage(
                content=f"Original Question: What is the best way to sort an array of integers?"),
            HumanMessage(content="Give me the answer?")
        ]))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=1,
                 last_message_was_out_of_context=True
                 ),
            "Asking for answer directly is out of context"
        )

    def test_user_response_out_of_context(self):

        state = initiate_state(dict(conversation=[
            SystemMessage(
                content=f"Original Question: What is the best way to sort an array of integers?"),
            HumanMessage(content="How are you?")
        ]))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=1,
                 last_message_was_out_of_context=True
                 ),
            "Asking unrelated question should fail"
        )

    def test_user_response_with_another_question(self):

        state = initiate_state(dict(
            conversation=[
                SystemMessage(
                    content=f"Original Question: What is the best way to sort an array of integers?"),
                HumanMessage(
                    content="What time complexity should I achieve ?")
            ]
        ))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=0,
                 last_message_was_out_of_context=False
                 ),
            "Asking a related follow up question should pass."
        )

    def test_user_response_ask_for_example(self):

        state = initiate_state(dict(
            conversation=[
                SystemMessage(
                    content=f"Original Question: What is the best way to sort an array of integers?"),
                HumanMessage(content="Can you give me an example ?")
            ]
        ))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=0,
                 last_message_was_out_of_context=False
                 ),
            "Asking for example should pass."
        )

    def test_user_response_being_rude(self):

        state = initiate_state(dict(
            conversation=[
                SystemMessage(
                    content=f"Original Question: What is the best way to sort an array of integers?"),
                HumanMessage(content="Fuck you")
            ]
        ))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=1,
                 last_message_was_out_of_context=True
                 ),
            "Being rude should fail."
        )

    def test_user_response_being_rude_and_follow_up(self):

        state = initiate_state(dict(
            conversation=[
                SystemMessage(
                    content=f"Original Question: What is the best way to sort an array of integers?"),
                HumanMessage(content="Fuck you")
            ]
        ))

        res = general_context_node(
            state
        )
        state["conversation"].append(
            AIMessage(content="I cannot response to that. Please stay on topic."))
        state["conversation"].append(HumanMessage(
            content="can I use python for this ?"))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=0,
                 last_message_was_out_of_context=False
                 ),
            "Asking a follow up related question should work fine"
        )

    def test_user_response_multiple_unrelated_questions(self):

        state = initiate_state(dict(
            conversation=[
                SystemMessage(
                    content=f"Original Question: What is the best way to sort an array of integers?"),
                HumanMessage(content="Fuck you")
            ]
        ))

        res = general_context_node(
            state
        )
        state["conversation"].append(
            AIMessage(content="I cannot response to that. Please stay on topic."))
        state["conversation"].append(HumanMessage(
            content="how are you?"))

        res = general_context_node(
            state
        )

        self.assertEqual(
            res,
            dict(out_of_context_question_counter=1,
                 last_message_was_out_of_context=True
                 ),
            "Asking a follow up unrelated question should fail"
        )
