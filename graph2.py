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



# -------- GRAPH -------- #
# checkpointer = MemorySaver()
graph = (
    StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    .add_node("create_task", generate_task)
    .add_edge(START, "create_task")
    .add_edge("create_task", END)
    .compile()
)