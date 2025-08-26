from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.config import get_store
from langgraph.types import Command, Send
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, InjectedStore, InjectedState
from langgraph_supervisor import create_supervisor
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

import redis
import json

from typing import Optional, Literal, List, Annotated, Dict
from typing_extensions import TypedDict

# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123



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



def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None):

    @tool(name, description=description)
    def handoff_tool(
        # The payload is the input to the handoff tool, matching your schema
        payload: Annotated[StoragePayload | RealtimePayload, "Payload for handoff"],
        # Inject current state and tool call id (ignored by LLM)
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        # Create a user message with the payload serialized as JSON or string
        task_description_message = {
            "role": "user",
            "content": f"Task for {agent_name}: {payload.model_dump_json()}"
        }
        # Prepare the input state for the next agent, replacing messages with the task description
        agent_input = {**state, "messages": [task_description_message]}
        # Return a Command that sends control to the target agent with the prepared input
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool

handoff_to_storage = create_custom_handoff_tool(agent_name="store_agent", name="handoff_to_storage", description="Delegate to storage agent")
handoff_to_realtime = create_custom_handoff_tool(agent_name="command_agent", name="handoff_to_realtime", description="Delegate to realtime agent")



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
    """Start real-time tracking of a face.

    Publishes a 'track' command to the Redis channel 'realtime_commands'.
    """
        
    payload = {"type": "track", "data": {"target": target}}
    if name:
        payload["data"]["name"] = name
    redis_client.publish("realtime_commands", json.dumps(payload))
    return f"Tracking started for {target} with name {name}"

@tool
def stop_tracking(target: str) -> str:
    """Stop real-time tracking of a face.

    Publishes a 'stop' command to the Redis channel 'realtime_commands'.
    """

    payload = {"type": "stop", "data": {"target": target}}
    redis_client.publish("realtime_commands", json.dumps(payload))
    return f"Stopped tracking {target}"

@tool
def send_rename(target: str, new_name: str) -> str:
    """Send a real-time rename command for a tracked face.

    Publishes a 'rename' command to the Redis channel 'realtime_commands'.
    """
    
    payload = {"type": "rename", "data": {"target": target, "new_name": new_name}}
    redis_client.publish("realtime_commands", json.dumps(payload))
    return f"Sent rename command for {target} to {new_name}"



# Create agents
store_agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[store_key, delete_key, get_key, rename_key],
    name="store_agent",
    store=get_store()
)

command_agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[track_face, stop_tracking, send_rename],
    name="command_agent",
    store=get_store()
)

# Create supervisor
supervisor_prompt = """
You are a supervisor agent responsible for routing user requests to specialized agents.

Your job is to:
- Understand the user's intent.
- Choose the correct handoff tool (handoff_to_storage or handoff_to_realtime).
- Construct the handoff tool arguments precisely according to their schemas.
- Delegate all work to the sub-agents; do not perform any task yourself.

Routing guidelines:
- If the user wants to store or manage embeddings (e.g., "store", "add", "save embedding", "rename X to Y"), use handoff_to_storage with the appropriate action and parameters.
- If the user wants to track or manage real-time face tracking (e.g., "track", "focus", "follow", "stop tracking"), use handoff_to_realtime with the appropriate action and parameters.

Always fill the handoff tool arguments exactly as per their schemas and do not attempt to do any work yourself.
"""


supervisor = create_supervisor(
    agents=[store_agent, command_agent],
    model=ChatOpenAI(model="gpt-4o"),
    tools=[handoff_to_storage, handoff_to_realtime],
    prompt=supervisor_prompt,
    name="supervisor"
).compile()


# supervisor_agent = create_react_agent(
#     model="openai:gpt-4o",
#     tools=[handoff_to_storage, handoff_to_realtime],
#     prompt=supervisor_prompt,
#     name="supervisor"
# )

# # Define the multi-agent supervisor graph
# supervisor = (
#     StateGraph(MessagesState)
#     # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
#     .add_node(supervisor_agent, destinations=("store_agent", "command_agent", END))
#     .add_node(store_agent)
#     .add_node(command_agent)
#     .add_edge(START, "supervisor")
#     # always return back to the supervisor
#     .add_edge("store_agent", "supervisor")
#     .add_edge("command_agent", "supervisor")
#     .compile()
# )