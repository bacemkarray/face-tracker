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






def main():
    client = get_sync_client(url="http://localhost:8123")
    
    # namespaces = client.store.list_namespaces(prefix=["face_embeddings"])
    # item = client.store.search_items(["face_embeddings"])

    # print(namespaces)
    # print(item)
    graph_name = "research-project-agents"
    
    thread = client.threads.create()
    result = client.runs.create(
        thread["thread_id"],
        graph_name,
        input={"messages":[HumanMessage(content="hello just testing something don't worry about this invocation")]})
    print(f"Invocation successful.")

if __name__ == "__main__":
    main()