from langgraph_sdk import get_client
import asyncio

async def send_embedding(face_id: int, emb: list[float]):
    client = get_client(url="http://localhost:8123")
    await client.store.put_item(namespace=["face_embeddings"],
                                key=face_id,
                                value={"embedding": emb})






async def main():
    client = get_client(url="http://localhost:8123")
    # await client.store.put_item(["face_embeddings"],
    #                             key="bacem",
    #                             value={"embedding": [0.1, 0.2, 0.3, 0.4]})
    
    namespaces = await client.store.list_namespaces(prefix=["face_embeddings"])
    item = await client.store.get_item(["face_embeddings"],
                                       key="bacem")

    print(namespaces)
    print(item)


    

asyncio.run(main())