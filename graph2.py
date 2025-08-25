from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from typing import List, Literal, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.storage.redis import RedisStore

# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

# redis_store = RedisStore(redis_url="redis://langgraph-redis:6379")
# from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o").with_structured_output(method="json_mode")

# class State(MessagesState):
#     id: str
#     embedding: Optional[List[float]]



# tools = [store_key, delete_key, get_key, rename_key, redis_publisher]
# llm = ChatOpenAI(model="gpt-4o")
# llm_with_tools = llm.bind_tools(tools)

# sys_msg = SystemMessage(content="You are a node in a workflow tasked with " \
# "interacting with a memory storage in a desired way based on an input. The only exception is if the user asks to track somebody, " \
# "you must initiate a real time command via the redis publisher.")

# # LLM node: invokes the LLM with bound tools
# def llm_node(state: State):
#     # invoke the LLM with the current messages or input
#     response = llm_with_tools.invoke([sys_msg] + state["messages"])
#     return {"messages": [response]}

# # Build the graph
# graph = (
#     StateGraph(MessagesState)
#     .add_node("llm_node", llm_node)
#     .add_node("tools", ToolNode(tools))
#     .add_edge(START, "llm_node")
#     .add_conditional_edges("llm_node", tools_condition)
#     .add_edge("llm_node", END)
#     .add_edge("tools", "llm_node")
#     .compile()
# )