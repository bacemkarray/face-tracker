from langchain_core.tools import tool
from langgraph.config import get_store
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, InjectedStore, InjectedState
from langgraph_supervisor import create_supervisor
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

import redis
import json

from typing import Optional, Literal, List, Annotated, Dict
from typing_extensions import TypedDict, Any

# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123


# Storage handoff schema
class StoragePayload(BaseModel):
    action: Literal["store", "get", "rename", "delete"] = Field(..., description="Action to perform")
    face_id: Optional[str] = Field(None, description="Face ID (required for rename/get/delete)")
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
        state: Annotated[Any, InjectedState],
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
            goto=agent_name,
            graph=Command.PARENT,
            update=agent_input
        )

    return handoff_tool

handoff_to_storage = create_custom_handoff_tool(agent_name="store_agent", name="handoff_to_storage", description="Delegate to storage agent")
handoff_to_realtime = create_custom_handoff_tool(agent_name="command_agent", name="handoff_to_realtime", description="Delegate to realtime agent")



#TOOLS FOR STORAGE AGENT
@tool
def store_key(key: str, value: List, store: Annotated[BaseStore, InjectedStore()]):
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



# prompts
store_prompt = """
You are the storage handler. 
Your only job is to manage face embeddings in persistent storage. 
You can:
- store a new embedding under a given key
- retrieve an embedding by key
- rename a key
- delete a key
- verify if a face embedding already exists in storage

Always:
- Use the provided tools to complete the task, never improvise.
- Reply with a short confirmation message so your supervisor knows the outcome.
"""

command_prompt = """
You are the realtime command handler. 
Your only job is to interact with the live face-tracking system by sending Redis commands. 
You can:
- track a person by ID or name
- stop tracking a person
- rename a tracked person

Always:
- Use the provided tools to execute the command.
- Respond with a one-line confirmation of what you did.
- Never attempt to manage storage or reroute tasks; that is your supervisor’s job.
"""

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

You must ensure that important relevant information is communicated between the two agents. If an embedding is renamed in the storage, then the command_agent must be aware of this. 
It is pivotal that this communication exists, otherwise the system will collapse.

Always fill the handoff tool arguments exactly as per their schemas and do not attempt to do any work yourself.
If the user command is incomplete or ambiguous, do not delegate immediately. Instead, ask the user a clarifying question to get the missing information. 
For example, if the user asks to "rename this embedding" or "delete the embedding" without specifying a name, 
respond with "Which embedding would you like renamed? and "Could you clarify which embedding to delete?" respectively. 
"""


# agents
store_agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[store_key, delete_key, get_key, rename_key],
    name="store_agent",
    store=get_store(),
    prompt=store_prompt
)

command_agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[track_face, stop_tracking, send_rename],
    name="command_agent",
    store=get_store(),
    prompt=command_prompt
)

supervisor_agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[handoff_to_storage, handoff_to_realtime],
    name="supervisor",
    prompt=supervisor_prompt
)

# supervisor = create_supervisor(
#     state_schema=State,
#     agents=[store_agent, command_agent],
#     model=ChatOpenAI(model="gpt-4o"),
#     tools=[handoff_to_storage, handoff_to_realtime],
#     prompt=supervisor_prompt,
#     name="supervisor"
# ).compile()


# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # 'destinations' is only needed for visualization and doesn't affect runtime behavior
    .add_node(supervisor_agent, destinations=("store_agent", "command_agent"))
    .add_node(store_agent)
    .add_node(command_agent)
    .add_edge(START, "supervisor")
    .add_edge("store_agent", "supervisor")
    .add_edge("command_agent", "supervisor")
    .compile()
)