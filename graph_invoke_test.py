from langgraph_sdk import get_sync_client

def send_embedding(face_id: int, emb: list[float]):
    print(f"🧠 send_embedding CALLED for ID {face_id}")
    client = get_sync_client(url="http://localhost:8123")
    client.store.put_item(["face_embeddings"],
                                key=f"{face_id}",
                                value={"embedding": emb})
    print(f"✅ embedding STORED for ID {face_id}")






# def main():
#     client = get_sync_client(url="http://localhost:8123")
    
#     # namespaces = client.store.list_namespaces(prefix=["face_embeddings"])
#     item = client.store.search_items(["face_embeddings"])

#     # print(namespaces)
#     print(item)

# if __name__ == "__main__":
#     main()