from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph, MessagesState

# -------- SCHEMA -------- #
class Task(TypedDict):
    task: Literal["search", "track"]
    duration: float = None
    target: str = None

class TaskPlannerState(TypedDict):
    instruction: str
    tasks: Optional[List[Task]]  # Will hold the parsed list of tasks


llm = ChatOpenAI(model="gpt-4o").with_structured_output(method="json_mode")

prompt_template = """
You are a task planner for a robotic arm. Convert the following instruction into a list of task objects in JSON format.

Each task should follow this format:
{{
  "task": "search" or "track",
  "duration": optional float (seconds),
  "target": optional string (person label like "dad" or "unknown_3") or null if not applicable
}}

Examples:

Input: search for 10 seconds
Output: [
  {{"task": "search", "target": null, "duration": 5}},
]

Input: follow dad for 10 seconds
Output: [
  {{"task": "track", "target": "dad", "duration": 10}}
]

Input: {instruction}
Output:
"""


def generate_task(state : TaskPlannerState):
    instruction = state["instruction"]
    prompt = prompt_template.format(instruction=instruction)
    system_message = SystemMessage(content=prompt)
    response = llm.invoke([system_message])
    return {"tasks": response}


# Build graph
builder = StateGraph(TaskPlannerState)
builder.add_node("generate_tasks", generate_task)

builder.add_edge(START, "generate_tasks")

# Compile graph
graph = builder.compile()