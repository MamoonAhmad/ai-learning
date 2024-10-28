"""General context node"""
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from src.models import get_chat_model
from src.nodes.agent_state import AgentState

# Load environment variables
_ = load_dotenv()

model = get_chat_model()

GENERAL_CONTEXT_PROMPT = """
    You are a smart interview answer analyzer whose job is analyze the latest response from the user and tell
    whether the response from the user is related to a question or not.
    User will be presented with an interview question. User can ask question in response and answer the 
    question. Your job is just to check whether the response from the question is related to that particular
    question or not. Your answer should only be yes or no. Yes meaning that the response from the 
    question is related to the question. No means it is not related to the original question.

    Examples

    Original Question: How would you sort a 20TB file with an integer on every line?
    User Answer: Can you please elaborate the question ?
    Your answer: Yes
    Because User's response is related to the question

    Original Question: How would you sort a 20TB file with an integer on every line?
    User Answer: where is the integer stored on every line ?
    Your answer: Yes
    Because User's response is related to the question

    Original Question: How would you sort a 20TB file with an integer on every line?
    User Answer: Can you dance ?
    Your answer: No
    Because User's response is not related to the question

    Original Question: How would you sort a 20TB file with an integer on every line?
    User Answer: how would you solve it ?
    Your answer: No
    Because User's response is not related to the question

    Original Question: What is the fastest algorithm to sort an array of strings ?
    User Answer: What is the time complexity we want to achieve ?
    Your answer: Yes
    Because User's response is related to the question

    Original Question: What is the fastest algorithm to sort an array of strings ?
    User Answer: What language should I solve this in?
    Your answer: Yes
    Because User's response is related to the question. language is a very broad term, but in this
    context it means programming language.

"""


def general_context_node(state: AgentState):
    """
        Responsible for making sure response from the user is within the context 
        and is safe to move forward in the node chain.
        Returns False If a user response is irrelevant.
        Returns True if the response is relevant.
    """

    messages = [
        SystemMessage(GENERAL_CONTEXT_PROMPT)
    ] + state["conversation"]

    response = model.invoke(messages)
    analysis = str(response.content).strip()

    is_out_of_context = 0 if analysis == "Yes" or analysis == "yes" else 1

    return dict(
        out_of_context_question_counter=state["out_of_context_question_counter"] +
        is_out_of_context,
        last_message_was_out_of_context=bool(is_out_of_context)
    )
