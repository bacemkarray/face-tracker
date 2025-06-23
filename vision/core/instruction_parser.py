from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI
import os
import json


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# -------- SCHEMA -------- #
class Task(BaseModel):
    task: str
    duration: Optional[float] = None
    target: Optional[str] = None

prompt_template = """
You are a task planner for a robotic arm. Convert the following instruction into a list of task objects in JSON format.

Each task should follow this format:
{{
  "task": "scan" or "track",
  "duration": optional float (seconds),
  "target": optional string (person label like "dad" or "unknown_3") or null if not applicable
}}

Examples:

Input: scan for 10 seconds
Output: [
  {{"task": "scan", "target": null, "duration": 5}},
]

Input: follow dad for 10 seconds
Output: [
  {{"task": "track", "target": "dad", "duration": 10}}
]

Input: {instruction}
Output:
"""

# -------- MAIN FUNCTION -------- #
def parse_instruction(instruction: str) -> List[Task]:
    prompt = prompt_template.format(instruction=instruction)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a task parser. Always output valid JSON arrays without Markdown formatting."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
        return [Task(**instruction).model_dump() for instruction in parsed] # Task(**arg) unpacks the dictionary.
        # purpose of using Pydantic is to ensure that the output data is valid and we can enforce the data-types.

    except Exception as e:
        print("Failed to parse LLM output:", e)
        print("Raw content:", content)
        return []

# -------- CLI DEBUG -------- #
if __name__ == "__main__":
    while True:
        instr = input("Instruction: ")
        tasks = parse_instruction(instr)
        for t in tasks:
            print("✅ Parsed:", t)
            