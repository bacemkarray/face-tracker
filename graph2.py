from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from typing import List, Optional, Annotated
from langgraph.graph import StateGraph, MessagesState, START, END

from langchain_core.tools import tool

from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
from langgraph.store.base import BaseStore


# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

# redis_store = RedisStore(redis_url="redis://langgraph-redis:6379")
# from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o").with_structured_output(method="json_mode")

class State(MessagesState):
    id: str
    embedding: Optional[List[float]]

#TOOLS FOR STORAGE AGENT
@tool
def store_key(key, value, store: Annotated[BaseStore, InjectedStore()]):
    """Stores a key-value pair in the store.
    
    Args:
        key: The key
        value: The value associated with the key
    """
    namespace = ("face_embeddings",)
    store.put(namespace, key, value)
    return f"Stored face embedding of {key}"

@tool
def delete_key(key: str, store: Annotated[BaseStore, InjectedStore()]):
    """Deletes a key-value pair from the persistent storage. Accessed via the key.
    
    Args:
        key: The key to the value you want to delete
    """

    namespace = ("face_embeddings",)
    store.delete(namespace, key)
    return f"Deleted {key} from storage"

@tool
def get_key(key: str, store: Annotated[BaseStore, InjectedStore()]):
    """Retrieves a value from the persistent storage by searching via the key.
    
    Args:
        key: The key associated with the value
    """
    namespace = ("face_embeddings",)
    value = store.get(namespace, key)
    return value

@tool
def rename_key(old_key: str, new_key: str, store: Annotated[BaseStore, InjectedStore()]):
    """Renames a key in the persistent storage.
    
    Args:
        old_key: previous key name
        new_key: desired key_name
    """

    # Fetch old entry
    namespace = ("face_embeddings",)
    old_entry = store.get(namespace, old_key)
    if old_entry is None:
        return f"No entry found for {old_key}"
    # Put new entry with same value
    store.put(namespace, new_key, old_entry.value)
    # Delete old entry
    store.delete(namespace, old_key)
    return f"Renamed {old_key} to {new_key}" 

tools = [store_key, delete_key, get_key, rename_key]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a node in a workflow tasked with " \
"interacting with a memory storage in a desired way based on an input. The only exception is if the user asks to track somebody, " \
"you must initiate a real time command via the redis publisher.")

# LLM node: invokes the LLM with bound tools
def llm_node(state: State):
    # invoke the LLM with the current messages or input
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# Build the graph
graph = (
    StateGraph(MessagesState)
    .add_node("llm_node", llm_node)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "llm_node")
    .add_conditional_edges("llm_node", tools_condition)
    .add_edge("llm_node", END)
    .add_edge("tools", "llm_node")
    .compile()
)