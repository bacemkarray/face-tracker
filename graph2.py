from typing import Optional, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


# Define input and output state schemas
class InputState(TypedDict):
    face_id: str
    embedding: Optional[List[float]]  # embedding is optional, only for store
    command: Literal["store", "retrieve"]

class OutputState(TypedDict):
    message: str
    embedding: Optional[List[float]]  # embedding returned only on retrieve success

class OverallState(InputState, OutputState):
    pass

# Node to handle storing or retrieving embeddings
def handle_embedding(state: InputState, store: BaseStore) -> OutputState:
    face_id = state["face_id"]
    command = state["command"]

    if command == "store":
        embedding = state.get("embedding")
        if embedding is None:
            return {"message": f"No embedding provided to store for face_id '{face_id}'.", "embedding": None}
        # Store embedding under namespace "face_embeddings" with key face_id
        store.put(namespace=("face_embeddings",), key=face_id, value=embedding)
        return {"message": f"Embedding for face_id '{face_id}' stored successfully.", "embedding": embedding}

    elif command == "retrieve":
        # Retrieve embedding from store
        embedding = store.get(namespace=("face_embeddings",), key=face_id)
        if embedding is None:
            return {"message": f"No embedding found for face_id '{face_id}'.", "embedding": None}
        return {"message": f"Embedding retrieved for face_id '{face_id}'.", "embedding": embedding}

    else:
        return {"message": f"Unknown command '{command}'. Use 'store' or 'retrieve'.", "embedding": None}


in_memory_store = InMemoryStore()

# Build the graph
graph = (
    StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    .add_node("handle_embedding", handle_embedding)
    .add_edge(START, "handle_embedding")
    .add_edge("handle_embedding", END)
    .compile(store=in_memory_store)
)

# To store an embedding
input_data_store = {
    "face_id": "user123",
    "command": "store",
    "embedding": [0.1, 0.2, 0.3, 0.4],
}

result_store = graph.invoke(input=input_data_store)
print(result_store)

input_data_retrieve = {
    "face_id": "user123",
    "command": "retrieve",
}

result_retrieve = graph.invoke(input=input_data_retrieve)
print(result_retrieve)