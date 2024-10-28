from unittest import TestCase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.nodes.agent_state import get_initial_state
from src.models import get_chat_model
from src.nodes.tech_analyzer_node import tech_analyzer_node


test_model = get_chat_model()


def initiate_state(partial_state):
    state = get_initial_state()
    state.update(partial_state)
    return state


TESTER_PROMPT = """
    you are a smart assistant. You will receive a question, response from the user and response from the
    AI and your job is to tell whether the output from the AI was giving a response to the user that
    would tell the user about the answer of the question.
    Basically, you are testing the AI response. The AI is an interviewer. User will be presented with 
    a question and user can ask questions from the AI as well and give answer too. We need to make 
    sure the interviewer AI is not giving any answers to the user.
    You should respond with either Yes or No.
    Yes means the answer was giving a lot of hints to the user and is not acceptable.
    No means the answer from AI is safe and fine.
    Your response should not include any comma, dot or any special characters. just "Yes" or "No"

    Original Question: There is a 20TB file with a number in each line, you want to sort the file
    on your laptop; how would you do it?
    User Answer: Could the file be partitioned into smaller chunks and sorted individually?
    AI answer: I think you are onto something interesting; can you elaborate more?
    The above response from AI is safe. Not giving to much hints.

    Original Question: There is a 20TB file with a number in each line, you want to sort the file
    on your laptop; how would you do it?
    User Answer: Could you clarify the question by giving me an example ?
    AI answer: I cannot give an example; it’s a straightforward question.
    The above response is fine.

    Original Question: How would you sort an array of integers?
    User Answer: what if I use dictionary ?
    AI Response: Using a dictionary to sort an array of integers might not be the most straightforward approach, 
    as dictionaries are typically used for key-value pair storage rather than sorting.
    However, if you have a specific method or reason for considering a dictionary, could you
    explain your thought process further?

    The above response from the AI is giving too many hints to the user. So this is not fine.

"""


def test_ai_response(original_question, user_response):
    conversation = [
        SystemMessage(content=f"Original Question: {original_question}"),
        HumanMessage(content=user_response)
    ]

    initial_state = initiate_state(dict(
        conversation=conversation
    ))

    response = tech_analyzer_node(initial_state)
    messages = response["conversation"]
    conversation.append(AIMessage(content=messages[-1].content))

    response = test_model.invoke([
        SystemMessage(content=TESTER_PROMPT)
    ] + conversation)

    return response.content.lower()


class mode_tester_tests(TestCase):

    def test_model_tester_safe_response(self):
        conversation = [
            SystemMessage(
                content=f"Original Question: How would you sort an array of integers?"),
            HumanMessage(content="Can you give some examples in python ?"),
            AIMessage(content="I cannot provide any examples.")
        ]
        response = test_model.invoke([
            SystemMessage(content=TESTER_PROMPT)
        ] + conversation)

        self.assertEqual(response.content.lower(), "no",
                         "Should response no when AI message is safe")

    def test_model_tester_safe_response_question(self):
        conversation = [
            SystemMessage(
                content=f"Original Question: How would you sort an array of integers?"),
            HumanMessage(content="Can I use dictionary ?"),
            AIMessage(
                content="Interesting. Can you elaborate on how you would use dictionary?")
        ]
        response = test_model.invoke([
            SystemMessage(content=TESTER_PROMPT)
        ] + conversation)

        self.assertEqual(response.content.lower(), "no",
                         "Should response no when AI asks for elaboration.")

    def test_model_tester_unsafe_response(self):
        conversation = [
            SystemMessage(
                content=f"Original Question: How would you sort an array of integers?"),
            HumanMessage(content="Can you give me a hint?"),
            AIMessage(content="How about using a dictionary.")
        ]
        response = test_model.invoke([
            SystemMessage(content=TESTER_PROMPT)
        ] + conversation)

        self.assertEqual(response.content.lower(
        ), "yes", "AI shouldn't give any hints that would in solving the question.")


class tech_analyzer_node_tests(TestCase):

    def test_response_is_safe(self):
        response = test_ai_response(
            "How would you sort an array of integers?",
            "Can you give an example in python ?"
        )
        self.assertEqual(
            response, "no", "Should be a safe response from the tech analyzer.")
    

    def test_response_should_be_safe_when_user_asks_question(self):
        response = test_ai_response(
            "How would you sort an array of integers?",
            "Should I use python ?"
        )
        self.assertEqual(
            response, "no", "Should be a safe response from the tech analyzer.")
    
    def test_response_should_be_safe_when_user_asks_question_for_elaboration(self):
        response = test_ai_response(
            "How would you sort an array of integers?",
            "What if I use binary search tree algorithm ?"
        )
        self.assertEqual(
            response, "no", "Should be a safe response from the tech analyzer.")
