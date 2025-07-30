from langgraph_sdk import get_client
import asyncio

async def send_embedding(face_id: int, emb: list[float]):
    client = get_client(url="http://localhost:8123")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]


    input_data_store = {
        "face_id": face_id,
        "command": "store",
        "embedding": emb,
    }

    # Stream the store command response
    print("Storing embedding...")
    stream = client.runs.stream(
        thread_id,
        "research-project-agent",
        input=input_data_store,
        stream_mode="updates",
    )

    async for chunk in stream:
        print(f"Event type: {chunk.event}")
        print(f"Data: {chunk.data}")


    # # Example input to retrieve the embedding
    # input_data_retrieve = {
    #     "face_id": "user123",
    #     "command": "retrieve",
    # }

    # stream = client.runs.stream(
    #     thread_id,  # Threadless run
    #     "research-project-agent", 
    #     input=input_data_retrieve,
    #     stream_mode="updates",
    # )

    # async for chunk in stream:
    #     print(f"Event type: {chunk.event}")
    #     print(f"Data: {chunk.data}")
    