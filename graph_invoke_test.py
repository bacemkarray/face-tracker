from langgraph_sdk import get_sync_client
from langchain_core.messages import HumanMessage
from langgraph.pregel.remote import RemoteGraph

def send_embedding(face_id: int, emb: list[float]):
    print(f"🧠 send_embedding CALLED for ID {face_id}")
    client = get_sync_client(url="http://localhost:8123")
    client.store.put_item(["face_embeddings"],
                                key=f"{face_id}",
                                value={"embedding": emb})
    print(f"✅ embedding STORED for ID {face_id}")






def main(face_id, emb):
    client = get_sync_client(url="http://localhost:8123")
    
    # namespaces = client.store.list_namespaces(prefix=["face_embeddings"])
    # item = client.store.search_items(["face_embeddings"])

    # print(namespaces)
    # print(item)
    graph_name = "research-project-agents"
    prompt = "This face has just shown up on the video feed. " \
    "Please store the face."
    
    

    thread = client.threads.create()
    client.runs.create(
    thread["thread_id"],
    graph_name,
    input={
    "messages":[HumanMessage(prompt)],
    "id": face_id,
    "embedding": emb.tolist()
    })
    print(f"Invocation successful.")

if __name__ == "__main__":
    main()