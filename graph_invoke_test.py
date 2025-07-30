from langgraph_sdk import get_client
import asyncio

async def send_embedding(face_id: int, emb: list[float]):
    print(f"🧠 send_embedding CALLED for ID {face_id}")
    client = get_client(url="http://localhost:8123")
    await client.store.put_item(namespace=["face_embeddings"],
                                key=face_id,
                                value={"embedding": emb})
    print(f"✅ embedding STORED for ID {face_id}")






async def main():
    client = get_client(url="http://localhost:8123")
    # await client.store.put_item(["face_embeddings"],
    #                             key="bacem",
    #                             value={"embedding": [0.1, 0.2, 0.3, 0.4]})
    
    namespaces = await client.store.list_namespaces(prefix=["face_embeddings"])
    item = await client.store.search_items(["face_embeddings"])

    print(namespaces)
    print(item)


    

asyncio.run(main())