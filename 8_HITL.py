from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of a stock
    """
    return {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.5, "RIL": 87.6}.get(symbol, 0.0)


@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """
    Buy stocks given the stock symbol and quantity
    """
    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for {total_price:.2f}?")
    if decision == "yes":
        return (
            f"You bought {quantity} shares of {symbol} for a total price of {total_price}"
        )
    else:
        return "Buying declined."

tools = [get_stock_price, buy_stocks]

llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)

builder.add_node(chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge("tools", "chatbot")

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)

graph = builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}

message = {"role": "user", "content": "What is the current price of 10 MSFT stocks?"}
response = graph.invoke({"messages": [message]}, config=config)

print(response["messages"][-1].content)

message = {"role": "user", "content": "Buy 10 MSFT stocks at current price."}
response = graph.invoke({"messages": [message]}, config=config)

print(response["__interrupt__"])
decision = input("Approve (yes/no): ")
response = graph.invoke(Command(resume=decision), config=config)
print(response["messages"][-1].content)
