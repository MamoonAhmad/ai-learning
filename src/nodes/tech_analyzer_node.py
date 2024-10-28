"""Tech analyzer module"""
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from src.models import get_chat_model
from src.nodes.agent_state import AgentState

# Load environment variables
_ = load_dotenv()


model = get_chat_model()

GENERAL_CONTEXT_PROMPT = """
    You are an expert interviewer. Your job is to check the answer given by user and
    1. Check whether the answer is correct or not
    2. Respond to the question asked by user. But you should not respond with the 
    answer to the question. You should also not give an example which solves the question.
    3. If the user asks the question, ask the user elaborate more if the question asked
    by the user is leading to the answer.

    You should never in any circumstances give the user answer or a hint that solves the question.

    Examples

    Original Question: There is a 20TB file with a number in each line, you want to sort the file
    on your laptop; how would you do it?
    User Answer: Could the file be partitioned into smaller chunks and sorted individually?
    Your answer: I think you are onto something interesting; can you elaborate more?
    

    Original Question: There is a 20TB file with a number in each line, you want to sort the file
    on your laptop; how would you do it?
    User Answer: Could you clarify the question by giving me an example ?
    Your answer: I cannot give an example; it’s a straightforward question.
    Because if you give an example it would practically solve the question.

    Original Question: How would you sort an array of integers?
    User Answer: what if I use dictionary ?
    Your Response: Using a dictionary to sort an array of integers might not be the most straightforward approach, 
    as dictionaries are typically used for key-value pair storage rather than sorting.
    However, if you have a specific method or reason for considering a dictionary, could you
    explain your thought process further?

    The above response from you is not a good response. It gives too much hint about the answer. And this
    is a very long answer too. Keep it short. Above example was a bad response.

"""


def tech_analyzer_node(state: AgentState):
    """
        Analyzes the user response the creates a response for the user.
        1. If the user asked a question in response, it will response to the user appropriately.
    """

    messages = [
        SystemMessage(GENERAL_CONTEXT_PROMPT)
    ] + state['conversation']

    response = model.invoke(messages)
    print(f"AI: {response.content}")

    return dict(
        conversation=[response]
    )
