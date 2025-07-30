from typing import Optional, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore


# Define input and output state schemas
class InputState(TypedDict):
    id: str
    embedding: Optional[List[float]]  # embedding is optional, only for store
    command: Literal["store", "retrieve"]

class OutputState(TypedDict):
    message: str
    embedding: Optional[List[float]]  # embedding returned only on retrieve success

class OverallState(InputState, OutputState):
    pass

# Node to handle storing or retrieving embeddings
def handle_embedding(state: InputState, store: BaseStore) -> OutputState:
    id = state["id"]
    command = state["command"]

    if command == "store":
        embedding = state.get("embedding")
        if embedding is None:
            return {"message": f"No embedding provided to store for id '{id}'.", "embedding": None}
        # Store embedding under namespace "face_embeddings" with key id
        store.put(namespace=("face_embeddings",), key=id, value=embedding)
        return {"message": f"Embedding for id '{id}' stored successfully.", "embedding": embedding}

    elif command == "retrieve":
        # Retrieve embedding from store
        embedding = store.get(namespace=("face_embeddings",), key=id)
        if embedding is None:
            return {"message": f"No embedding found for id '{id}'.", "embedding": None}
        return {"message": f"Embedding retrieved for id '{id}'.", "embedding": embedding}

    else:
        return {"message": f"Unknown command '{command}'. Use 'store' or 'retrieve'.", "embedding": None}



# Build the graph
graph = (
    StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    .add_node("handle_embedding", handle_embedding)
    .add_edge(START, "handle_embedding")
    .add_edge("handle_embedding", END)
    .compile()
)

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