from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode, InjectedStore
from langgraph.store.base import BaseStore
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field

import redis
import json

from typing import Optional, Literal, List, Annotated
from typing_extensions import TypedDict



# class State(MessagesState):
#     id: str
#     embedding: Optional[List[float]]




# Storage handoff schema
class StoragePayload(BaseModel):
    action: Literal["store", "get", "rename", "delete"] = Field(..., description="Action to perform")
    face_id: Optional[str] = Field(None, description="Face ID (required for rename/get/delete)")
    embedding: Optional[list[float]] = Field(None, description="Embedding vector (required for store)")
    metadata: Optional[dict] = Field(None, description="Optional metadata")
    old_key: Optional[str] = Field(None, description="Old key (rename only)")
    new_key: Optional[str] = Field(None, description="New key (rename only)")

# Realtime handoff schema
class RealtimePayload(BaseModel):
    action: Literal["track", "stop", "rename"] = Field(..., description="Action to perform")
    target: str = Field(..., description="Face ID or name")
    new_name: Optional[str] = Field(None, description="New name (rename only)")


@tool("handoff_to_storage", args_schema=StoragePayload, return_direct=True)
def handoff_to_storage(payload: StoragePayload) -> str:
    # Translate handoff payload to storage tool calls
    """Route a storage-related request to the storage agent."""
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

@tool("handoff_to_realtime", args_schema=RealtimePayload, return_direct=True)
def handoff_to_realtime(payload: RealtimePayload) -> str:
    """Route a real-time tracking request to the realtime agent."""
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





#TOOLS FOR REALTIME AGENT

redis_client = redis.Redis(host="langgraph-redis", port=6379, db=0, decode_responses=True)

@tool
def track_face(target: str, name: str | None = None) -> str:
    payload = {"type": "track", "data": {"target": target}}
    if name:
        payload["data"]["name"] = name
    redis_client.publish("realtime_commands", json.dumps(payload))
    return f"Tracking started for {target} with name {name}"

@tool
def stop_tracking(target: str) -> str:
    payload = {"type": "stop", "data": {"target": target}}
    redis_client.publish("realtime_commands", json.dumps(payload))
    return f"Stopped tracking {target}"

@tool
def send_rename(target: str, new_name: str) -> str:
    payload = {"type": "rename", "data": {"target": target, "new_name": new_name}}
    redis_client.publish("realtime_commands", json.dumps(payload))
    return f"Sent rename command for {target} to {new_name}"





# Create agents
store_handler = create_react_agent(
    model="openai:gpt-4o",
    tools=[store_key, delete_key, get_key, rename_key],
    name="store_handler"
)

command_handler = create_react_agent(
    model="openai:gpt-4o",
    tools=[track_face, stop_tracking, send_rename],
    name="command_handler"
)


# Create supervisor
supervisor_prompt = """
You are a supervisor agent routing user requests to specialized agents.

Routing rules:
- If user says "store", "add", or "save embedding", call handoff_to_storage with action="store".
- If user says "rename X to Y", call handoff_to_storage with action="rename".
- If user says "track", "focus", or "follow" NAME or ID, call handoff_to_realtime with action="track".
- If user says "stop tracking", call handoff_to_realtime with action="stop".

Fill the handoff tool arguments exactly as per their schemas.

Do not do any work yourself; delegate all tasks to the sub-agents.
"""


supervisor = create_supervisor(
    agents=[store_handler, command_handler],
    model=ChatOpenAI(model="gpt-4o"),
    tools=[handoff_to_storage, handoff_to_realtime],
    prompt=supervisor_prompt,
    name="supervisor"
)







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