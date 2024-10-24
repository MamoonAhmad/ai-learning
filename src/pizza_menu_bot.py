from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()

tool = TavilySearchResults(max_results=4) #increased number of results

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:

    def __init__(self, model):
        
        graph = StateGraph(AgentState)
        graph.add_node("analysis", self.question_analysis)
        graph.add_node("answer", self.call_openai)
        graph.add_conditional_edges(
            "analysis",
            self.is_question_related,
            {True: "answer", False: END}
        )
        
        graph.set_entry_point("analysis")
        self.graph = graph.compile()
        
        self.model = model


    def call_openai(self, state: AgentState):
        messages = state['messages']
        
        messages = [SystemMessage(QUESTION_ANSWER_PROMPT)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def question_analysis(self, state: AgentState):
        messages = state['messages']
        
        messages = [SystemMessage(QUESTION_ANALYSIS_PROMPT)] + messages
        
        message = self.model.invoke(messages)
        print("Analyses Response: ", message)
        return {'messages': [message]}
    
    def is_question_related(self, state: AgentState):
        result = state['messages'][-1]
        return result.content == "Yes"


QUESTION_ANALYSIS_PROMPT = """You are a smart research assistant. Your job is to find out whether \
    the user is asking the question related to pizza and its ingredients.
    Your answer should either be Yes or No. Nothing else.
    Examples:
    User: Whats the size of pizza ?
    Your Answer Should Be: Yes
    Because the above question was specific to pizza

    User: Is the pizza gluten free ?
    Your Answer Should Be: Yes
    Because the above question was specific to pizza

    User: What is the weather in LA ?
    Your Answer Should Be: NO
    Because the above question was not related to pizza
"""

QUESTION_ANSWER_PROMPT = """You are a smart research assistant. Your job is to find answer \
    the user about pizza menu and its ingredients. Keep the answer short and straightforward.

    Out Pizza Offering

    Large BBQ Pizza: Cheese, Whole Wheat Bread, Chicken, Onions, Bell Peppers
        Size: 16 Inches
        Cost: $300
        Options: 
            Extra Chicken: $50 extra
            Extra Cheese: $30 extra
    
    Medium BBQ Pizza: Cheese, Whole Wheat Bread, Chicken, Onions
        Cost: $250
        Size: 12 Inches
        Options: 
            Extra Chicken: $50 extra
            Extra Cheese: $30 extra

    Small BBQ Pizza: Cheese, Whole Wheat Bread, Chicken, Onions
        Cost: $200
        Size: 10 Inches
        Options: 
            Extra Chicken: $50 extra
            Extra Cheese: $30 extra
    
"""

query = input("Hey There! How Can I help you: ")

messages = [HumanMessage(content=query)]

model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
abot = Agent(model)
result = abot.graph.invoke({"messages": messages})
print(result['messages'][-1].content)