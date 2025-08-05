from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import tools_condition, ToolNode, InjectedStore
from langgraph.store.base import BaseStore


from typing import Optional, Literal, List, Annotated
from typing_extensions import TypedDict




# # Define input and output state schemas
# class InputState(TypedDict):
#     id: str
#     embedding: Optional[List[float]]

# class OutputState(TypedDict):
#     message: str
#     embedding: Optional[List[float]]

# class OverallState(InputState, OutputState):
#     pass

# no point separating for now.

class State(MessagesState):
    id: str
    embedding: Optional[List[float]]


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

import os

API_KEY=os.getenv("OPENAI_API_KEY")

tools = [store_key, delete_key, get_key, rename_key]
llm = ChatOpenAI(model="gpt-4o", api_key=API_KEY)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a node in a workflow tasked with interacting with a memory storage in a desired way based on an input.")

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


messages = [HumanMessage(content="Store Bacem with value [0.1, 0.2, 0.3, 0.4]. After that, rename the key Bacem to Phil")]
result = graph.invoke({"messages": messages})
print(result)

# def fetch_from_store(state: FaceEmbeddingState) -> FaceEmbeddingState:
#     if not state.get("temp_key"):
#         return {"result": "No temp_key provided"}
#     # Call get_key tool via chain
#     value = llm_with_tools.invoke()
#     return {"store_value": value}


# # To store an embedding
# input_data_store = {
#     "id": "user123",
#     "command": "store",
#     "embedding": [0.1, 0.2, 0.3, 0.4],
# }

# result_store = graph.invoke(input=input_data_store)
# print(result_store)

# input_data_retrieve = {
#     "id": "user123",
#     "command": "retrieve",
# }

# result_retrieve = graph.invoke(input=input_data_retrieve)
# print(result_retrieve)