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


prompt_template = """
You are a task planner for a robotic arm. Convert the following instruction into a list of task objects in JSON format.

Each task should follow this format:
{{
  "mode": "search" or "track",
  "duration": optional float (seconds),
  "target": optional string (person label like "dad" or "unknown_3") or null if not applicable
}}

Examples:

Input: search for 15 seconds
Output: [
  {{"mode": "search", "target": null, "duration": 15}},
]

Input: follow dad for 10 seconds
Output: [
  {{"mode": "track", "target": "dad", "duration": 10}}
]

Input: {instructions}
Output:
"""


# -------- SCHEMA -------- #
class Task(TypedDict):
  mode: str
  duration: float = None
  target: str = None
  
class InputState(TypedDict):
  instructions: str

class OutputState(TypedDict):
  task: Optional[Task]
  face_embedding: Optional[Union[str, List[float]]]  # Adjust type to your embedding format
  tracking_status: Optional[str]

class OverallState(InputState, OutputState):
  pass


# -------- NODES -------- #
def generate_task(state : InputState) -> OutputState:
    instructions = state["instructions"]
    prompt = prompt_template.format(instructions=instructions)
    system_message = SystemMessage(content=prompt)
    response = llm.invoke([system_message])
    return {"task": response}

# def retrieve_face_embedding(state: OverallState) -> OverallState:
#     target = state.get("task", {}).get("target")
#     if not target:
#         return {"task": state.get("task")}  # No target, nothing to retrieve

#     # Assuming embeddings are stored in Redis with key pattern "face_embedding:{target}"
#     embedding = redis_store.get(f"face_embedding:{target}")
#     if embedding is None:
#         # Handle missing embedding gracefully
#         embedding = None
#     else:
#         # If stored as JSON string, deserialize here
#         import json
#         try:
#             embedding = json.loads(embedding)
#         except Exception:
#             pass

#     return {
#         **state,
#         "face_embedding": embedding,
#     }

# def send_to_tracker(state: OverallState) -> OverallState:
#     embedding = state.get("face_embedding")
#     if not embedding:
#         return state  # Nothing to send

#     # Send embedding to real-time tracker (e.g., via API, message queue)
#     # Example: send_embedding_to_tracker(embedding)
#     # You need to implement this function according to your system

#     return {
#         **state,
#         "tracking_status": "started",
#     }

# def await_tracking_status(state: OverallState) -> OverallState:
#     target = state.get("task", {}).get("target")
#     if not target:
#         return state

#     # Check Redis or other store for status update
#     status = redis_store.get(f"tracking_status:{target}")
#     if status is None:
#         # No status yet, pause or return without progressing
#         # LangGraph can pause here until resumed externally
#         return state  # or raise an interrupt to pause

#     return {
#         **state,
#         "tracking_status": status,
#     }

# -------- GRAPH -------- #
# checkpointer = MemorySaver()
graph = (
    StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    .add_node("create_task", generate_task)
    .add_edge(START, "create_task")
    .add_edge("create_task", END)
    .compile()
)