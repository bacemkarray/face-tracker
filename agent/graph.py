from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode, InjectedStore
from langgraph.store.base import BaseStore
from langgraph_supervisor import create_supervisor

import redis
import json

from typing import Optional, Literal, List, Annotated
from typing_extensions import TypedDict



class State(MessagesState):
    id: str
    embedding: Optional[List[float]]









from pydantic import BaseModel, Field
from typing import Literal, Optional


# Storage handoff schema

class StoragePayload(BaseModel):
    action: Literal["store", "get", "rename", "delete"] = Field(..., description="Action to perform")
    face_id: Optional[str] = Field(None, description="Face ID (required for rename/get/delete)")
    embedding: Optional[list[float]] = Field(None, description="Embedding vector (required for store)")
    metadata: Optional[dict] = Field(None, description="Optional metadata")
    old_key: Optional[str] = Field(None, description="Old key (rename only)")
    new_key: Optional[str] = Field(None, description="New key (rename only)")


@tool("handoff_to_storage", args_schema=StoragePayload, return_direct=True)
def handoff_to_storage(payload: StoragePayload) -> str:
    # Translate handoff payload to storage tool calls
    p = payload
    if p.action == "store":
        return store_key.invoke({"face_id": p.face_id, "embedding": p.embedding, "metadata": p.metadata})
    elif p.action == "get":
        return get_key.invoke({"key": p.face_id})
    elif p.action == "rename":
        return rename_key.invoke({"old_key": p.old_key, "new_key": p.new_key})
    elif p.action == "delete":
        return delete_key.invoke({"key": p.face_id})
    else:
        return f"Unknown storage action: {p.action}"

# Realtime handoff schema

class RealtimePayload(BaseModel):
    action: Literal["track", "stop", "rename"] = Field(..., description="Action to perform")
    target: str = Field(..., description="Face ID or name")
    new_name: Optional[str] = Field(None, description="New name (rename only)")

@tool("handoff_to_realtime", args_schema=RealtimePayload, return_direct=True)
def handoff_to_realtime(payload: RealtimePayload) -> str:
    p = payload
    if p.action == "track":
        return track_face.invoke({"target": p.target})
    elif p.action == "stop":
        return stop_tracking.invoke({"target": p.target})
    elif p.action == "rename":
        if p.new_name is None:
            return "Error: new_name is required for rename action"
        return send_rename.invoke({"target": p.target, "new_name": p.new_name})
    else:
        return f"Unknown realtime action: {p.action}"







































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







@tool
def redis_publisher(key, value):
    """Sends a tracking command to the real-time code."""
    client = redis.Redis(host="langgraph-redis", port=6379, db=0, decode_responses=True)

    message = {"data": key}
    payload = json.dumps(message)
    client.publish("realtime_commands", payload)
    return f"Published {message} to real-time code"











store_handler = create_react_agent(
    model="openai:gpt-4o",
    tools=[store_key, delete_key, get_key, rename_key],
    prompt="You are a flight booking assistant",
    name="store_handler"
)

command_handler = create_react_agent(
    model="openai:gpt-4o",
    tools=[redis_publisher],
    prompt="You are a flight booking assistant",
    name="store_handler"
)




supervisor = create_supervisor(
    agents=[store_handler, command_handler],
    model=ChatOpenAI(model="gpt-4o"),
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    )
).compile()






























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