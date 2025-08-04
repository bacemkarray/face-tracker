from typing import Optional, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langchain_core.tools import tool


# Define input and output state schemas
class InputState(TypedDict):
    id: str
    embedding: Optional[List[float]]  # embedding is optional, only for store

class OutputState(TypedDict):
    message: str
    embedding: Optional[List[float]]  # embedding returned only on retrieve success

class OverallState(InputState, OutputState):
    pass





@tool
def rename_key(old_key: str, new_key: str, store: BaseStore):
    # Fetch old entry
    old_entry = store.get(("face_embeddings"), old_key)
    if old_entry is None:
        return f"No entry found for {old_key}"
    # Put new entry with same value
    store.put(("face_embeddings",), new_key, old_entry.value)
    # Delete old entry
    store.delete(("face_embeddings"), old_key)
    return f"Renamed {old_key} to {new_key}"


@tool
def store_key(key, value, store: BaseStore):
    store.put(("face_embeddings",), key, value)
    return f"Stored face embedding of {key}"


@tool
def get_key(key, store: BaseStore):
    value = store.get(("face_embeddings",), key)
    return value





# Node to handle storing or retrieving embeddings
def handle_embedding(state: InputState, store: BaseStore) -> OutputState:
    id = state["id"]

    embedding = state.get("embedding")
    if embedding is None:
        return {"message": f"No embedding provided to store for id '{id}'.", "embedding": None}
    # Store embedding under namespace "face_embeddings" with key id
    store.put(namespace=("face_embeddings",), key=id, value=embedding)
    return {"message": f"Embedding for id '{id}' stored successfully.", "embedding": embedding}

    # Retrieve embedding from store
    embedding = store.get(namespace=("face_embeddings",), key=id)
    if embedding is None:
        return {"message": f"No embedding found for id '{id}'.", "embedding": None}
    return {"message": f"Embedding retrieved for id '{id}'.", "embedding": embedding}

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